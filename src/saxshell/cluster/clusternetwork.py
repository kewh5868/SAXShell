from __future__ import annotations

import json
import re
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from string import ascii_uppercase
from time import monotonic

import numpy as np
from scipy.spatial import KDTree

from saxshell.mdtrajectory.frame.manager import TrajectoryManager
from saxshell.structure import (
    AtomTypeDefinitions,
    PDBAtom,
    PDBStructure,
    normalize_atom_type_definitions,
)

PairCutoffDefinitions = dict[tuple[str, str], dict[int, float]]
SEARCH_MODE_KDTREE = "kdtree"
SEARCH_MODE_VECTORIZED = "vectorized"
SEARCH_MODE_BRUTEFORCE = "bruteforce"
SEARCH_MODE_CHOICES = (
    SEARCH_MODE_KDTREE,
    SEARCH_MODE_VECTORIZED,
    SEARCH_MODE_BRUTEFORCE,
)
CLUSTER_METADATA_FILENAME = "cluster_extraction_metadata.json"
DEFAULT_SAVE_STATE_FREQUENCY = 1000
FRAME_STATUS_PROCESSED = "processed"
FRAME_STATUS_COMPLETED = "completed"
METADATA_CHECKPOINT_SECONDS = 5.0
FRAME_FOLDER_LABELS = {
    "pdb": "PDB frames",
    "xyz": "XYZ frames",
}
FRAME_FOLDER_EXTENSIONS = {
    "pdb": ".pdb",
    "xyz": ".xyz",
}
_PBC_FILENAME_PATTERN = re.compile(
    r"_pbc_([0-9pPxX]+)",
    flags=re.IGNORECASE,
)
_FRAME_INDEX_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def normalize_search_mode(search_mode: str | None) -> str:
    """Normalize cluster neighbor-search mode names."""
    if search_mode is None:
        return SEARCH_MODE_KDTREE

    normalized = (
        str(search_mode).strip().lower().replace("-", "").replace("_", "")
    )
    if normalized in {"auto", "fast", "tree", "kdtree"}:
        return SEARCH_MODE_KDTREE
    if normalized in {"numpy", "vector", "vectorized"}:
        return SEARCH_MODE_VECTORIZED
    if normalized in {"brute", "bruteforce"}:
        return SEARCH_MODE_BRUTEFORCE
    raise ValueError(
        "Unsupported search mode. Choose 'kdtree', 'vectorized', or "
        "'bruteforce'."
    )


def format_search_mode_label(search_mode: str) -> str:
    """Return a user-facing label for one search mode."""
    normalized = normalize_search_mode(search_mode)
    if normalized == SEARCH_MODE_KDTREE:
        return "KDTree"
    if normalized == SEARCH_MODE_VECTORIZED:
        return "Vectorized"
    return "Brute force"


def normalize_save_state_frequency(value: int | None) -> int:
    """Normalize the frame count used for metadata checkpoints."""
    if value is None:
        return DEFAULT_SAVE_STATE_FREQUENCY

    normalized = int(value)
    if normalized <= 0:
        raise ValueError(
            "Save-state frequency must be a positive whole number of "
            "frames."
        )
    return normalized


def _elements_by_atom_type(
    atom_types: Mapping[int, str],
    elements: Mapping[int, str],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for atom_id, atom_type in atom_types.items():
        grouped.setdefault(atom_type, set()).add(elements[atom_id].title())
    return grouped


def _query_positions_for_cutoff(
    reference_position: np.ndarray,
    cutoff: float,
    *,
    box: np.ndarray | None,
    use_pbc: bool,
) -> list[np.ndarray]:
    """Return KDTree query centers for one point under optional PBC."""
    if not use_pbc or box is None or cutoff <= 0.0:
        return [reference_position]

    shift_options: list[tuple[int, ...]] = []
    for coordinate, axis_length in zip(reference_position, box):
        if axis_length <= 0.0:
            shift_options.append((0,))
            continue

        axis_shifts = [0]
        if coordinate <= cutoff:
            axis_shifts.append(1)
        if axis_length - coordinate <= cutoff:
            axis_shifts.append(-1)
        shift_options.append(tuple(axis_shifts))

    query_positions: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for shifts in product(*shift_options):
        shifted_position = reference_position + (
            np.asarray(shifts, dtype=float) * box
        )
        key = tuple(float(value) for value in shifted_position)
        if key in seen:
            continue
        seen.add(key)
        query_positions.append(shifted_position)
    return query_positions


def _vectorized_deltas(
    reference_position: np.ndarray,
    coordinates: np.ndarray,
    *,
    box: np.ndarray | None,
    use_pbc: bool,
) -> np.ndarray:
    """Return coordinate deltas for one reference point."""
    deltas = coordinates - reference_position
    if use_pbc and box is not None:
        positive_axes = box > 0.0
        if np.any(positive_axes):
            deltas = deltas.copy()
            deltas[:, positive_axes] -= (
                np.round(deltas[:, positive_axes] / box[positive_axes])
                * box[positive_axes]
            )
    return deltas


def _utc_timestamp() -> str:
    """Return one UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _metadata_file_path(output_dir: Path) -> Path:
    """Return the metadata filename used for one cluster export."""
    return output_dir / CLUSTER_METADATA_FILENAME


def _frame_entry_is_processed(entry: Mapping[str, object]) -> bool:
    """Return whether one frame entry has finished cluster
    extraction."""
    return str(entry.get("status")) in {
        FRAME_STATUS_PROCESSED,
        FRAME_STATUS_COMPLETED,
    }


def _frame_entry_is_completed(entry: Mapping[str, object]) -> bool:
    """Return whether one frame entry has finished the final sort
    step."""
    return str(entry.get("status")) == FRAME_STATUS_COMPLETED


def _write_json_file(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON atomically to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _frame_manifest_entry(path: Path) -> dict[str, object]:
    """Return one lightweight file manifest entry for signature
    checks."""
    stat_result = path.stat()
    return {
        "name": path.name,
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _serialize_atom_type_definitions(
    atom_type_definitions: AtomTypeDefinitions,
) -> dict[str, list[list[str | None]]]:
    """Return a JSON-serializable atom-type definition mapping."""
    serialized: dict[str, list[list[str | None]]] = {}
    for atom_type, rules in normalize_atom_type_definitions(
        atom_type_definitions
    ).items():
        serialized[atom_type] = [
            [element, residue]
            for element, residue in sorted(
                rules,
                key=lambda item: (item[0], "" if item[1] is None else item[1]),
            )
        ]
    return serialized


def _serialize_pair_cutoff_definitions(
    pair_cutoffs_def: Mapping[
        tuple[str, str],
        Mapping[int, float] | float | int,
    ],
) -> dict[str, dict[str, float]]:
    """Return a JSON-serializable pair-cutoff mapping."""
    serialized: dict[str, dict[str, float]] = {}
    for pair, levels in sorted(
        normalize_pair_cutoffs(pair_cutoffs_def).items(),
        key=lambda item: item[0],
    ):
        serialized[f"{pair[0]}:{pair[1]}"] = {
            str(level): float(cutoff)
            for level, cutoff in sorted(levels.items())
        }
    return serialized


def _relative_output_paths(
    output_dir: Path,
    paths: Sequence[Path],
) -> list[str]:
    """Return output-file paths relative to the export directory."""
    return sorted(path.relative_to(output_dir).as_posix() for path in paths)


def _absolute_output_paths(
    output_dir: Path,
    relative_paths: Sequence[str],
) -> list[Path]:
    """Resolve relative metadata paths back to absolute paths."""
    return [output_dir / relative_path for relative_path in relative_paths]


def _entry_sort_key(entry: Mapping[str, object]) -> tuple[int, str]:
    """Return a stable sort key for stored frame metadata entries."""
    return (
        int(entry.get("frame_index", 0)),
        str(entry.get("frame_name", "")),
    )


def normalize_pair_cutoffs(
    pair_cutoffs: Mapping[
        tuple[str, str],
        Mapping[int, float] | float | int,
    ],
) -> PairCutoffDefinitions:
    """Normalize pair cutoffs to title-cased element keys and int
    levels."""
    normalized: PairCutoffDefinitions = {}
    for pair, values in pair_cutoffs.items():
        normalized_pair = (pair[0].title(), pair[1].title())
        if isinstance(values, Mapping):
            normalized_levels = {
                int(level): float(cutoff) for level, cutoff in values.items()
            }
        else:
            normalized_levels = {0: float(values)}
        normalized[normalized_pair] = normalized_levels
    return normalized


def frame_folder_label(frame_format: str) -> str:
    """Return the UI label for one extracted frame-set format."""
    return FRAME_FOLDER_LABELS.get(frame_format, frame_format.upper())


def frame_output_suffix(frame_format: str) -> str:
    """Return the cluster-export file suffix for one frame-set
    format."""
    return FRAME_FOLDER_EXTENSIONS.get(frame_format, f".{frame_format}")


def _frame_sort_key(path: Path) -> tuple[int, int, str]:
    """Return a natural sort key for extracted frame filenames."""
    match = _FRAME_INDEX_PATTERN.search(path.stem)
    if match is None:
        return (1, 0, path.name.lower())
    return (0, int(match.group(1)), path.name.lower())


def detect_frame_folder_mode(
    frames_dir: str | Path,
) -> tuple[str, list[Path]]:
    """Detect whether an extracted frame folder contains PDB or XYZ
    files."""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise ValueError(
            f"The selected frames folder does not exist: {frames_dir}"
        )

    discovered = {
        frame_format: sorted(
            (
                path
                for path in frames_dir.iterdir()
                if path.suffix.lower() == suffix
            ),
            key=_frame_sort_key,
        )
        for frame_format, suffix in FRAME_FOLDER_EXTENSIONS.items()
    }
    present_formats = [
        (frame_format, paths)
        for frame_format, paths in discovered.items()
        if paths
    ]

    if not present_formats:
        raise ValueError(
            "The selected folder does not contain any .pdb or .xyz frame "
            "files."
        )
    if len(present_formats) > 1:
        raise ValueError(
            "The selected folder mixes .pdb and .xyz frame files. Please "
            "choose a folder containing only one extracted frame format."
        )

    return present_formats[0]


def estimate_box_dimensions_from_coordinates(
    coordinates: Sequence[np.ndarray] | np.ndarray,
) -> tuple[float, float, float] | None:
    """Estimate box lengths from coordinate extents."""
    coordinate_array = np.asarray(coordinates, dtype=float)
    if coordinate_array.size == 0:
        return None
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
        raise ValueError("Coordinate arrays must have shape (n_atoms, 3).")
    extents = np.ptp(coordinate_array, axis=0)
    return tuple(float(value) for value in extents)


def parse_pbc_box_token(
    token: str,
) -> tuple[float, float, float] | None:
    """Parse one ``_pbc_...`` filename token into box dimensions."""
    normalized = token.strip().lower()
    if not normalized:
        return None

    parts = [part for part in normalized.split("x") if part]
    if not parts:
        return None

    values: list[float] = []
    for part in parts:
        if not re.fullmatch(r"\d+(?:p\d+)?", part):
            return None
        values.append(float(part.replace("p", ".")))

    if len(values) == 1:
        value = values[0]
        return (value, value, value)
    if len(values) == 3:
        return tuple(values)
    return None


def detect_source_box_dimensions(
    frames_dir: str | Path,
) -> tuple[tuple[float, float, float], Path] | None:
    """Look for a sibling ``*-pos-1.xyz`` file with ``_pbc_``
    metadata."""
    frames_dir = Path(frames_dir)
    parent_dir = frames_dir.parent
    if not parent_dir.is_dir():
        return None

    for path in sorted(parent_dir.glob("*-pos-1.xyz")):
        match = _PBC_FILENAME_PATTERN.search(path.name)
        if match is None:
            continue
        box_dimensions = parse_pbc_box_token(match.group(1))
        if box_dimensions is not None:
            return box_dimensions, path
    return None


def stoichiometry_label(stoichiometry: Mapping[str, int]) -> str:
    """Return a stable folder label for one cluster stoichiometry."""
    counts = {
        str(element).title(): int(count)
        for element, count in stoichiometry.items()
        if int(count) > 0
    }
    if not counts:
        return "Unassigned"

    pieces: list[str] = []
    for preferred in ("Pb", "I"):
        if preferred in counts:
            count = counts.pop(preferred)
            pieces.append(preferred if count == 1 else f"{preferred}{count}")

    for element in sorted(counts):
        count = counts[element]
        pieces.append(element if count == 1 else f"{element}{count}")

    return "".join(pieces)


def _move_cluster_files_to_stoichiometry_dirs(
    output_dir: Path,
    frame_dir: Path,
    frame_label: str,
    clusters: Sequence["ClusterRecord"],
    *,
    suffix: str,
    stoichiometry_dir_cache: dict[str, Path] | None = None,
) -> list[Path]:
    """Move temporary frame-grouped files into stoichiometry folders."""
    moved_files: list[Path] = []

    for cluster in clusters:
        temp_path = frame_dir / f"{frame_label}_{cluster.cluster_id}{suffix}"
        stoich_label = stoichiometry_label(cluster.stoichiometry)
        if stoichiometry_dir_cache is None:
            stoich_dir = output_dir / stoich_label
            stoich_dir.mkdir(parents=True, exist_ok=True)
        else:
            stoich_dir = stoichiometry_dir_cache.get(stoich_label)
            if stoich_dir is None:
                stoich_dir = output_dir / stoich_label
                stoich_dir.mkdir(parents=True, exist_ok=True)
                stoichiometry_dir_cache[stoich_label] = stoich_dir
        final_path = stoich_dir / temp_path.name
        if temp_path.exists():
            temp_path.replace(final_path)
            moved_files.append(final_path)
            continue
        if final_path.exists():
            moved_files.append(final_path)

    if frame_dir.is_dir() and not any(frame_dir.iterdir()):
        frame_dir.rmdir()

    return moved_files


@dataclass(slots=True)
class ClusterRecord:
    """Cluster membership summary for one frame."""

    cluster_id: str
    atom_ids: tuple[int, ...]
    solute_atom_ids: tuple[int, ...]
    node_atom_ids: tuple[int, ...]
    linker_atom_ids: tuple[int, ...]
    shell_atom_ids: tuple[int, ...]
    stoichiometry: dict[str, int]
    shell_levels: dict[int, int | str | None]

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "atom_ids": list(self.atom_ids),
            "solute_atom_ids": list(self.solute_atom_ids),
            "node_atom_ids": list(self.node_atom_ids),
            "linker_atom_ids": list(self.linker_atom_ids),
            "shell_atom_ids": list(self.shell_atom_ids),
            "stoichiometry": dict(self.stoichiometry),
            "shell_levels": dict(self.shell_levels),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "ClusterRecord":
        """Build one record from JSON-compatible metadata."""
        shell_levels = {
            int(atom_id): value
            for atom_id, value in dict(payload.get("shell_levels", {})).items()
        }
        return cls(
            cluster_id=str(payload["cluster_id"]),
            atom_ids=tuple(
                int(value) for value in payload.get("atom_ids", [])
            ),
            solute_atom_ids=tuple(
                int(value) for value in payload.get("solute_atom_ids", [])
            ),
            node_atom_ids=tuple(
                int(value) for value in payload.get("node_atom_ids", [])
            ),
            linker_atom_ids=tuple(
                int(value) for value in payload.get("linker_atom_ids", [])
            ),
            shell_atom_ids=tuple(
                int(value) for value in payload.get("shell_atom_ids", [])
            ),
            stoichiometry={
                str(element): int(count)
                for element, count in dict(
                    payload.get("stoichiometry", {})
                ).items()
            },
            shell_levels=shell_levels,
        )


@dataclass(slots=True)
class FrameClusterResult:
    """Cluster-analysis output for one trajectory frame."""

    frame_index: int
    time_fs: float | None
    clusters: list[ClusterRecord]

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "time_fs": self.time_fs,
            "n_clusters": self.n_clusters,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "FrameClusterResult":
        """Build one frame result from JSON-compatible metadata."""
        time_fs = payload.get("time_fs")
        return cls(
            frame_index=int(payload["frame_index"]),
            time_fs=None if time_fs is None else float(time_fs),
            clusters=[
                ClusterRecord.from_dict(cluster_payload)
                for cluster_payload in payload.get("clusters", [])
            ],
        )


@dataclass(slots=True)
class TrajectoryClusterExport:
    """Written cluster files produced from a trajectory analysis."""

    written_files: list[Path]
    frame_results: list[FrameClusterResult]
    metadata_path: Path | None = None
    resumed: bool = False
    already_complete: bool = False
    previously_completed_frames: int = 0
    newly_processed_frames: int = 0


@dataclass(slots=True)
class XYZAtom:
    """Structured representation of one XYZ atom record."""

    atom_id: int
    element: str
    coordinates: np.ndarray
    cluster_id: str | None = None
    atom_type: str = "unassigned"
    shell_level: int | str | None = None
    shell_history: list[tuple[str, int | str | None]] = field(
        default_factory=list
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

    def copy(self) -> "XYZAtom":
        """Return a copy suitable for edited output coordinates."""
        return XYZAtom(
            atom_id=self.atom_id,
            element=self.element,
            coordinates=self.coordinates.copy(),
            cluster_id=self.cluster_id,
            atom_type=self.atom_type,
            shell_level=self.shell_level,
            shell_history=list(self.shell_history),
        )


class XYZStructure:
    """XYZ atom container with element-based atom-type assignment."""

    def __init__(
        self,
        filepath: str | Path | None = None,
        atom_type_definitions: AtomTypeDefinitions | None = None,
        atoms: list[XYZAtom] | None = None,
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
        self.atoms: list[XYZAtom] = list(atoms) if atoms is not None else []

        if self.filepath is not None and not self.atoms:
            self.read_xyz_file()
        elif self.atom_type_definitions:
            self.assign_atom_types()

    @classmethod
    def from_lines(
        cls,
        lines: Sequence[str],
        atom_type_definitions: AtomTypeDefinitions | None = None,
        *,
        source_name: str | None = None,
    ) -> "XYZStructure":
        """Build a structure from already-loaded XYZ lines."""
        structure = cls(
            atom_type_definitions=atom_type_definitions,
            atoms=[],
            source_name=source_name,
        )
        structure.read_xyz_lines(lines)
        return structure

    def read_xyz_file(self) -> None:
        """Load atoms from ``self.filepath``."""
        if self.filepath is None:
            raise ValueError("No XYZ filepath was provided.")
        try:
            with self.filepath.open("r") as handle:
                self.read_xyz_lines(handle.readlines())
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"XYZ file not found at path: {self.filepath}"
            ) from exc

    def read_xyz_lines(self, lines: Sequence[str]) -> None:
        """Parse atoms from one XYZ frame."""
        if not lines:
            raise ValueError("The XYZ frame is empty.")

        try:
            atom_count = int(lines[0].strip())
        except ValueError as exc:
            raise ValueError(
                "The XYZ frame does not start with an atom count."
            ) from exc

        atom_lines = [line for line in lines[2:] if line.strip()]
        if len(atom_lines) < atom_count:
            raise ValueError(
                "The XYZ frame does not contain enough atom lines for the "
                "declared atom count."
            )

        self.atoms = []
        for atom_index, line in enumerate(atom_lines[:atom_count], start=1):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ atom line: {line!r}")
            element = "".join(
                character for character in parts[0] if character.isalpha()
            ).title()
            if not element:
                raise ValueError(f"Could not infer an element from {line!r}")
            coordinates = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=float,
            )
            atom = XYZAtom(
                atom_id=atom_index,
                element=element,
                coordinates=coordinates,
            )
            if self.atom_type_definitions:
                self.assign_atom_type(atom)
            self.atoms.append(atom)

    def assign_atom_type(self, atom: XYZAtom) -> None:
        """Assign atom types using only element matching."""
        atom.atom_type = "unassigned"
        atom.shell_level = None

        atom_element = atom.element.title()
        for atom_type, criteria_list in self.atom_type_definitions.items():
            for element, _residue_name in criteria_list:
                if atom_element == element.title():
                    atom.atom_type = atom_type
                    atom.shell_level = "free" if atom_type == "shell" else None
                    return

    def assign_atom_types(self) -> None:
        """Apply atom-type definitions to every atom."""
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
        dict[int, XYZAtom],
        dict[int, XYZAtom],
    ]:
        """Return atom metadata maps keyed by atom id."""
        if include_atom_types is None:
            include_atom_types = {"node", "linker", "shell"}

        atom_ids_list: list[int] = []
        coordinates_list: list[np.ndarray] = []
        elements: dict[int, str] = {}
        atom_types: dict[int, str] = {}
        atom_id_map: dict[int, XYZAtom] = {}
        all_atoms_map: dict[int, XYZAtom] = {}

        for atom in self.atoms:
            atom_id = atom.atom_id
            all_atoms_map[atom_id] = atom
            if atom.atom_type not in include_atom_types:
                continue
            atom_ids_list.append(atom_id)
            coordinates_list.append(atom.coordinates.copy())
            elements[atom_id] = atom.element
            atom_types[atom_id] = atom.atom_type
            atom_id_map[atom_id] = atom

        return (
            atom_ids_list,
            coordinates_list,
            elements,
            atom_types,
            atom_id_map,
            all_atoms_map,
        )

    def write_xyz_file(
        self,
        output_path: str | Path,
        atoms: list[XYZAtom] | None = None,
        *,
        comment: str | None = None,
    ) -> Path:
        """Write atoms to an XYZ file and return the output path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atoms_to_write = sorted(
            atoms if atoms is not None else self.atoms,
            key=lambda atom: atom.atom_id,
        )

        with output_path.open("w") as handle:
            handle.write(f"{len(atoms_to_write)}\n")
            handle.write(
                f"{comment or 'XYZ cluster extracted from frame data'}\n"
            )
            for atom in atoms_to_write:
                handle.write(
                    f"{atom.element} "
                    f"{atom.coordinates[0]:.6f} "
                    f"{atom.coordinates[1]:.6f} "
                    f"{atom.coordinates[2]:.6f}\n"
                )
        return output_path


class ClusterNetwork:
    """Find solute cluster networks within one PDB structure."""

    def __init__(
        self,
        pdb_structure: PDBStructure,
        atom_type_definitions: AtomTypeDefinitions,
        pair_cutoffs_def: Mapping[
            tuple[str, str],
            Mapping[int, float] | float | int,
        ],
        *,
        box_dimensions: Sequence[float] | None = None,
        default_cutoff: float | None = None,
        use_pbc: bool = False,
        include_shell_atoms_in_stoichiometry: bool = False,
        search_mode: str = SEARCH_MODE_KDTREE,
    ) -> None:
        self.pdb_structure = pdb_structure
        self.atom_type_definitions = atom_type_definitions
        self.pair_cutoffs_def = normalize_pair_cutoffs(pair_cutoffs_def)
        self.default_cutoff = default_cutoff
        self.use_pbc = bool(use_pbc)
        self.include_shell_atoms_in_stoichiometry = bool(
            include_shell_atoms_in_stoichiometry
        )
        self.search_mode = normalize_search_mode(search_mode)
        self.box = (
            np.asarray(box_dimensions, dtype=float)
            if box_dimensions is not None
            else None
        )

        (
            self.atom_ids_list,
            coordinates_list,
            self.elements,
            self.atom_types,
            self.residue_names,
            self.atom_id_map,
            self.all_atoms_map,
        ) = self.pdb_structure.get_atoms_data()

        self.base_atom_types = dict(self.atom_types)
        self.original_coordinates = np.array(coordinates_list, dtype=float)
        self.kd_tree = (
            KDTree(self.original_coordinates)
            if len(self.original_coordinates)
            else None
        )
        self.atom_index_map = {
            atom_id: index for index, atom_id in enumerate(self.atom_ids_list)
        }
        self._elements_by_base_atom_type = _elements_by_atom_type(
            self.base_atom_types,
            self.elements,
        )
        self.cluster_ids: set[str] = set()
        self.reserved_ids: set[str] = set()
        self._reset_state()

    def _reset_state(self) -> None:
        self.atom_types = dict(self.base_atom_types)
        self.found_status = {atom_id: False for atom_id in self.atom_ids_list}
        self.shell_levels = {atom_id: None for atom_id in self.atom_ids_list}
        self.clusters: list[set[int]] = []
        self.cluster_labels: list[str] = []
        self.atom_virtual_positions: dict[int, np.ndarray] = {}

        for atom in self.atom_id_map.values():
            atom.cluster_id = None
            atom.shell_level = "free" if atom.atom_type == "shell" else None
            atom.shell_history.clear()

    def _pair_cutoff(
        self,
        first_element: str,
        second_element: str,
        shell_level: int,
    ) -> float | None:
        pair = (first_element.title(), second_element.title())
        reverse_pair = (pair[1], pair[0])

        levels = self.pair_cutoffs_def.get(pair)
        if levels is not None and shell_level in levels:
            return levels[shell_level]

        reverse_levels = self.pair_cutoffs_def.get(reverse_pair)
        if reverse_levels is not None and shell_level in reverse_levels:
            return reverse_levels[shell_level]

        return self.default_cutoff

    def _max_cutoff_for_allowed_types(
        self,
        parent_element: str,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
    ) -> float | None:
        candidate_elements: set[str] = set()
        for atom_type in allowed_atom_types:
            candidate_elements.update(
                self._elements_by_base_atom_type.get(atom_type, set())
            )

        max_cutoff: float | None = None
        for candidate_element in candidate_elements:
            cutoff = self._pair_cutoff(
                parent_element,
                candidate_element,
                shell_level,
            )
            if cutoff is None or cutoff <= 0.0:
                continue
            if max_cutoff is None or cutoff > max_cutoff:
                max_cutoff = cutoff
        return max_cutoff

    def _minimum_image_delta(
        self,
        reference_position: np.ndarray,
        other_position: np.ndarray,
    ) -> np.ndarray:
        delta = other_position - reference_position
        if self.use_pbc and self.box is not None:
            positive_axes = self.box > 0.0
            if np.any(positive_axes):
                delta = delta.copy()
                delta[positive_axes] -= (
                    np.round(delta[positive_axes] / self.box[positive_axes])
                    * self.box[positive_axes]
                )
        return delta

    def pbc_distance(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
    ) -> float:
        """Return the minimum-image distance between two positions."""
        return float(np.linalg.norm(self._minimum_image_delta(pos1, pos2)))

    def search_atoms_within_pair_cutoff(
        self,
        parent_id: int,
        parent_virtual_pos: np.ndarray,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
        allow_reuse: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        """Return neighbor atoms matched by the cutoff table."""
        if self.search_mode == SEARCH_MODE_KDTREE:
            return self._search_atoms_within_pair_cutoff_kdtree(
                parent_id,
                parent_virtual_pos,
                shell_level,
                allowed_atom_types=allowed_atom_types,
                allow_reuse=allow_reuse,
            )
        if self.search_mode == SEARCH_MODE_VECTORIZED:
            return self._search_atoms_within_pair_cutoff_vectorized(
                parent_id,
                parent_virtual_pos,
                shell_level,
                allowed_atom_types=allowed_atom_types,
                allow_reuse=allow_reuse,
            )
        if self.search_mode == SEARCH_MODE_VECTORIZED:
            return self._search_atoms_within_pair_cutoff_vectorized(
                parent_id,
                parent_virtual_pos,
                shell_level,
                allowed_atom_types=allowed_atom_types,
                allow_reuse=allow_reuse,
            )
        return self._search_atoms_within_pair_cutoff_bruteforce(
            parent_id,
            parent_virtual_pos,
            shell_level,
            allowed_atom_types=allowed_atom_types,
            allow_reuse=allow_reuse,
        )

    def _search_atoms_within_pair_cutoff_bruteforce(
        self,
        parent_id: int,
        parent_virtual_pos: np.ndarray,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
        allow_reuse: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        parent_index = self.atom_index_map[parent_id]
        parent_real_pos = self.original_coordinates[parent_index]
        parent_element = self.elements[parent_id]
        results: list[tuple[int, np.ndarray]] = []

        for neighbor_id in self.atom_ids_list:
            if neighbor_id == parent_id:
                continue
            if self.atom_types[neighbor_id] not in allowed_atom_types:
                continue
            if not allow_reuse and self.found_status[neighbor_id]:
                continue

            cutoff = self._pair_cutoff(
                parent_element,
                self.elements[neighbor_id],
                shell_level,
            )
            if cutoff is None or cutoff <= 0.0:
                continue

            neighbor_real_pos = self.original_coordinates[
                self.atom_index_map[neighbor_id]
            ]
            delta = self._minimum_image_delta(
                parent_real_pos, neighbor_real_pos
            )
            if float(np.linalg.norm(delta)) > cutoff:
                continue
            results.append((neighbor_id, parent_virtual_pos + delta))

        return results

    def _search_atoms_within_pair_cutoff_kdtree(
        self,
        parent_id: int,
        parent_virtual_pos: np.ndarray,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
        allow_reuse: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        if self.kd_tree is None:
            return []

        parent_index = self.atom_index_map[parent_id]
        parent_real_pos = self.original_coordinates[parent_index]
        parent_element = self.elements[parent_id]
        max_cutoff = self._max_cutoff_for_allowed_types(
            parent_element,
            shell_level,
            allowed_atom_types=allowed_atom_types,
        )
        if max_cutoff is None or max_cutoff <= 0.0:
            return []

        candidate_indices: set[int] = set()
        for query_position in _query_positions_for_cutoff(
            parent_real_pos,
            max_cutoff,
            box=self.box,
            use_pbc=self.use_pbc,
        ):
            candidate_indices.update(
                int(index)
                for index in self.kd_tree.query_ball_point(
                    query_position,
                    r=max_cutoff,
                )
            )

        results: list[tuple[int, np.ndarray]] = []
        for neighbor_index in candidate_indices:
            neighbor_id = self.atom_ids_list[neighbor_index]
            if neighbor_id == parent_id:
                continue
            if self.atom_types[neighbor_id] not in allowed_atom_types:
                continue
            if not allow_reuse and self.found_status[neighbor_id]:
                continue

            cutoff = self._pair_cutoff(
                parent_element,
                self.elements[neighbor_id],
                shell_level,
            )
            if cutoff is None or cutoff <= 0.0:
                continue

            neighbor_real_pos = self.original_coordinates[neighbor_index]
            delta = self._minimum_image_delta(
                parent_real_pos,
                neighbor_real_pos,
            )
            if float(np.linalg.norm(delta)) > cutoff:
                continue
            results.append((neighbor_id, parent_virtual_pos + delta))

        return results

    def _search_atoms_within_pair_cutoff_vectorized(
        self,
        parent_id: int,
        parent_virtual_pos: np.ndarray,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
        allow_reuse: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        parent_index = self.atom_index_map[parent_id]
        parent_real_pos = self.original_coordinates[parent_index]
        parent_element = self.elements[parent_id]

        atom_type_mask = np.fromiter(
            (
                self.atom_types[atom_id] in allowed_atom_types
                for atom_id in self.atom_ids_list
            ),
            dtype=bool,
            count=len(self.atom_ids_list),
        )
        atom_type_mask[parent_index] = False
        if not allow_reuse:
            found_mask = np.fromiter(
                (self.found_status[atom_id] for atom_id in self.atom_ids_list),
                dtype=bool,
                count=len(self.atom_ids_list),
            )
            atom_type_mask &= ~found_mask
        if not np.any(atom_type_mask):
            return []

        cutoff_values = np.fromiter(
            (
                self._pair_cutoff(
                    parent_element,
                    self.elements[atom_id],
                    shell_level,
                )
                or 0.0
                for atom_id in self.atom_ids_list
            ),
            dtype=float,
            count=len(self.atom_ids_list),
        )
        candidate_mask = atom_type_mask & (cutoff_values > 0.0)
        if not np.any(candidate_mask):
            return []

        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_positions = self.original_coordinates[candidate_indices]
        deltas = _vectorized_deltas(
            parent_real_pos,
            candidate_positions,
            box=self.box,
            use_pbc=self.use_pbc,
        )
        distances = np.linalg.norm(deltas, axis=1)
        within_cutoff = distances <= cutoff_values[candidate_indices]
        if not np.any(within_cutoff):
            return []

        results: list[tuple[int, np.ndarray]] = []
        for neighbor_index, delta in zip(
            candidate_indices[within_cutoff],
            deltas[within_cutoff],
        ):
            neighbor_id = self.atom_ids_list[int(neighbor_index)]
            results.append((neighbor_id, parent_virtual_pos + delta))
        return results

    def generate_cluster_id(self) -> str:
        """Generate a unique three-letter cluster identifier."""
        max_ids = 26**3
        for value in range(max_ids):
            temp = value
            cluster_id = ""
            for _ in range(3):
                cluster_id = ascii_uppercase[temp % 26] + cluster_id
                temp //= 26
            if (
                cluster_id not in self.cluster_ids
                and cluster_id not in self.reserved_ids
            ):
                self.cluster_ids.add(cluster_id)
                return cluster_id
        raise ValueError("All possible cluster IDs are reserved or in use.")

    def assign_cluster_id_and_shell_level(
        self,
        cluster: set[int],
        cluster_id: str,
    ) -> None:
        """Apply cluster metadata to the atoms contained in one
        cluster."""
        for atom_id in cluster:
            atom = self.atom_id_map[atom_id]
            atom.update_cluster_assignment(
                cluster_id,
                shell_level=self.shell_levels[atom_id],
            )

    def create_cluster(self, start_id: int) -> set[int]:
        """Build a connected solute cluster starting from one seed
        atom."""
        current_cluster: set[int] = set()
        start_index = self.atom_index_map[start_id]
        start_position = self.original_coordinates[start_index]
        self.atom_virtual_positions[start_id] = start_position

        search_stack: deque[tuple[int, np.ndarray]] = deque(
            [(start_id, start_position)]
        )
        self.found_status[start_id] = True
        self.shell_levels[start_id] = 0
        current_cluster.add(start_id)

        while search_stack:
            current_id, current_virt_pos = search_stack.pop()
            for (
                neighbor_id,
                neighbor_virtual_pos,
            ) in self.search_atoms_within_pair_cutoff(
                parent_id=current_id,
                parent_virtual_pos=current_virt_pos,
                shell_level=0,
                allowed_atom_types={"node", "linker", "shell"},
            ):
                if self.found_status[neighbor_id]:
                    continue
                self.found_status[neighbor_id] = True
                self.shell_levels[neighbor_id] = 0
                self.atom_virtual_positions[neighbor_id] = neighbor_virtual_pos
                current_cluster.add(neighbor_id)
                if self.base_atom_types[neighbor_id] != "shell":
                    search_stack.append((neighbor_id, neighbor_virtual_pos))

        return current_cluster

    def process_shell_level_for_cluster(
        self,
        cluster: set[int],
        cluster_id: str,
        shell_level: int,
        *,
        shared_shells: bool = False,
    ) -> None:
        """Add shell atoms at a given shell level to one cluster."""
        node_ids = [
            atom_id
            for atom_id in cluster
            if self.base_atom_types.get(atom_id) == "node"
        ]
        new_shells: dict[int, np.ndarray] = {}

        for node_id in node_ids:
            parent_virtual_pos = self.atom_virtual_positions[node_id]
            for (
                neighbor_id,
                neighbor_virtual_pos,
            ) in self.search_atoms_within_pair_cutoff(
                parent_id=node_id,
                parent_virtual_pos=parent_virtual_pos,
                shell_level=shell_level,
                allowed_atom_types={"shell"},
                allow_reuse=shared_shells,
            ):
                if neighbor_id not in new_shells:
                    new_shells[neighbor_id] = neighbor_virtual_pos

        for neighbor_id, neighbor_virtual_pos in new_shells.items():
            if neighbor_id in cluster:
                continue
            if not shared_shells and self.found_status[neighbor_id]:
                continue
            if not shared_shells:
                self.found_status[neighbor_id] = True
            cluster.add(neighbor_id)
            self.atom_types[neighbor_id] = "shell"
            self.shell_levels[neighbor_id] = shell_level
            self.atom_virtual_positions[neighbor_id] = neighbor_virtual_pos
            self.atom_id_map[neighbor_id].update_cluster_assignment(
                cluster_id,
                shell_level=shell_level,
            )

    def find_clusters(
        self,
        *,
        shell_levels: Sequence[int] = (1, 2),
        shared_shells: bool = False,
    ) -> list[ClusterRecord]:
        """Find cluster networks for the current structure."""
        self._reset_state()

        for atom_id, atom_type in self.base_atom_types.items():
            if atom_type == "node" and not self.found_status[atom_id]:
                cluster = self.create_cluster(atom_id)
                cluster_id = self.generate_cluster_id()
                self.assign_cluster_id_and_shell_level(cluster, cluster_id)
                self.clusters.append(cluster)
                self.cluster_labels.append(cluster_id)

        for atom_id, atom_type in self.base_atom_types.items():
            if atom_type == "linker" and not self.found_status[atom_id]:
                cluster = self.create_cluster(atom_id)
                cluster_id = self.generate_cluster_id()
                self.assign_cluster_id_and_shell_level(cluster, cluster_id)
                self.clusters.append(cluster)
                self.cluster_labels.append(cluster_id)

        for shell_level in shell_levels:
            if shell_level <= 0:
                continue
            for cluster, cluster_id in zip(self.clusters, self.cluster_labels):
                self.process_shell_level_for_cluster(
                    cluster,
                    cluster_id,
                    shell_level,
                    shared_shells=shared_shells,
                )

        return self.cluster_records()

    def gather_complete_molecules_for_cluster(
        self,
        original_cluster_atom_ids: set[int],
        shell_atom_ids: set[int],
    ) -> set[int]:
        """Return cluster atom ids plus complete residues for shell
        atoms."""
        shell_residues = {
            (
                self.atom_id_map[atom_id].residue_number,
                self.atom_id_map[atom_id].residue_name,
            )
            for atom_id in shell_atom_ids
        }

        complete_ids = set(original_cluster_atom_ids)
        for atom in self.pdb_structure.atoms:
            residue_info = (atom.residue_number, atom.residue_name)
            if residue_info in shell_residues:
                complete_ids.add(atom.atom_id)
        return complete_ids

    def identify_wrapped_atoms(
        self,
        atom_ids: set[int],
    ) -> dict[int, dict[str, object]]:
        """Report atoms whose virtual positions differ by box
        translations."""
        wrapped: dict[int, dict[str, object]] = {}
        if not self.use_pbc or self.box is None:
            return wrapped

        for atom_id in atom_ids:
            virtual_position = self.atom_virtual_positions.get(atom_id)
            if virtual_position is None:
                continue
            if atom_id not in self.atom_index_map:
                continue
            real_position = self.original_coordinates[
                self.atom_index_map[atom_id]
            ]
            shift = np.round(
                (virtual_position - real_position) / self.box
            ).astype(int)
            if np.any(shift):
                wrapped[atom_id] = {
                    "shift": shift,
                    "real": real_position,
                    "virtual": virtual_position,
                    "residue": self.atom_id_map[atom_id].residue_number,
                }
        return wrapped

    def build_cluster_atoms(
        self,
        cluster_index: int,
        *,
        include_shell_levels: Sequence[int] = (0, 1, 2),
    ) -> list[PDBAtom]:
        """Build output atoms for one cluster using virtual
        coordinates."""
        cluster = self.clusters[cluster_index]
        core_ids = {
            atom_id
            for atom_id in cluster
            if self.shell_levels.get(atom_id) == 0
        }
        shell_ids = {
            atom_id
            for atom_id in cluster
            if self.atom_types.get(atom_id) == "shell"
            and self.shell_levels.get(atom_id) in include_shell_levels
        }

        original_ids = set(core_ids)
        if any(level > 0 for level in include_shell_levels):
            original_ids |= shell_ids
        if not original_ids:
            return []

        complete_ids = self.gather_complete_molecules_for_cluster(
            original_ids,
            shell_ids,
        )
        final_atoms: list[PDBAtom] = []
        for atom_id in sorted(complete_ids):
            atom = self.all_atoms_map.get(atom_id)
            if atom is None:
                continue
            copied_atom = atom.copy()
            virtual_position = self.atom_virtual_positions.get(atom_id)
            if virtual_position is not None:
                copied_atom.coordinates = virtual_position.copy()
            final_atoms.append(copied_atom)

        wrapped_atoms = self.identify_wrapped_atoms(complete_ids)
        residue_shifts: dict[int, np.ndarray] = {}
        for info in wrapped_atoms.values():
            residue = int(info["residue"])
            shift = tuple(int(value) for value in info["shift"])
            residue_shifts.setdefault(residue, []).append(np.array(shift))

        majority_shifts: dict[int, np.ndarray] = {}
        for residue, shifts in residue_shifts.items():
            majority_shift, _ = Counter(
                tuple(shift) for shift in shifts
            ).most_common(1)[0]
            majority_shifts[residue] = np.array(majority_shift, dtype=int)

        if self.box is not None:
            for atom in final_atoms:
                shift = majority_shifts.get(atom.residue_number)
                if shift is None or not np.any(shift):
                    continue
                if atom.atom_id in wrapped_atoms:
                    continue
                atom.coordinates = atom.coordinates + shift * self.box

        return final_atoms

    def write_cluster_pdb_files(
        self,
        output_dir: str | Path,
        *,
        frame_label: str | None = None,
        include_shell_levels: Sequence[int] = (0, 1, 2),
    ) -> list[Path]:
        """Write per-cluster PDB files for the current structure."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written_files: list[Path] = []
        source_label = (
            frame_label or self.pdb_structure.source_name or "cluster_frame"
        )

        for cluster_index, cluster_id in enumerate(self.cluster_labels):
            atoms = self.build_cluster_atoms(
                cluster_index,
                include_shell_levels=include_shell_levels,
            )
            if not atoms:
                continue
            output_path = output_dir / f"{source_label}_{cluster_id}.pdb"
            self.pdb_structure.write_pdb_file(output_path, atoms)
            written_files.append(output_path)

        return written_files

    def cluster_records(self) -> list[ClusterRecord]:
        """Return cluster summary records for the current structure."""
        records: list[ClusterRecord] = []
        for cluster, cluster_id in zip(self.clusters, self.cluster_labels):
            atom_ids = tuple(sorted(cluster))
            node_atom_ids = tuple(
                atom_id
                for atom_id in atom_ids
                if self.base_atom_types.get(atom_id) == "node"
            )
            linker_atom_ids = tuple(
                atom_id
                for atom_id in atom_ids
                if self.base_atom_types.get(atom_id) == "linker"
            )
            shell_atom_ids = tuple(
                atom_id
                for atom_id in atom_ids
                if self.atom_types.get(atom_id) == "shell"
                and atom_id not in node_atom_ids
                and atom_id not in linker_atom_ids
            )
            solute_atom_ids = tuple(sorted(node_atom_ids + linker_atom_ids))
            stoichiometry_atom_ids = list(solute_atom_ids)
            if self.include_shell_atoms_in_stoichiometry:
                stoichiometry_atom_ids.extend(shell_atom_ids)
            stoichiometry = dict(
                Counter(
                    self.elements[atom_id]
                    for atom_id in stoichiometry_atom_ids
                )
            )
            shell_level_map = {
                atom_id: self.shell_levels.get(atom_id) for atom_id in atom_ids
            }
            records.append(
                ClusterRecord(
                    cluster_id=cluster_id,
                    atom_ids=atom_ids,
                    solute_atom_ids=solute_atom_ids,
                    node_atom_ids=node_atom_ids,
                    linker_atom_ids=linker_atom_ids,
                    shell_atom_ids=shell_atom_ids,
                    stoichiometry=stoichiometry,
                    shell_levels=shell_level_map,
                )
            )
        return records


class XYZClusterNetwork:
    """Find solute cluster networks within one XYZ structure."""

    def __init__(
        self,
        xyz_structure: XYZStructure,
        atom_type_definitions: AtomTypeDefinitions,
        pair_cutoffs_def: Mapping[
            tuple[str, str],
            Mapping[int, float] | float | int,
        ],
        *,
        box_dimensions: Sequence[float] | None = None,
        default_cutoff: float | None = None,
        use_pbc: bool = False,
        include_shell_atoms_in_stoichiometry: bool = False,
        search_mode: str = SEARCH_MODE_KDTREE,
    ) -> None:
        self.xyz_structure = xyz_structure
        self.atom_type_definitions = atom_type_definitions
        self.pair_cutoffs_def = normalize_pair_cutoffs(pair_cutoffs_def)
        self.default_cutoff = default_cutoff
        self.use_pbc = bool(use_pbc)
        self.include_shell_atoms_in_stoichiometry = bool(
            include_shell_atoms_in_stoichiometry
        )
        self.search_mode = normalize_search_mode(search_mode)
        self.box = (
            np.asarray(box_dimensions, dtype=float)
            if box_dimensions is not None
            else None
        )

        (
            self.atom_ids_list,
            coordinates_list,
            self.elements,
            self.atom_types,
            self.atom_id_map,
            self.all_atoms_map,
        ) = self.xyz_structure.get_atoms_data()

        self.base_atom_types = dict(self.atom_types)
        self.original_coordinates = np.array(coordinates_list, dtype=float)
        self.kd_tree = (
            KDTree(self.original_coordinates)
            if len(self.original_coordinates)
            else None
        )
        self.atom_index_map = {
            atom_id: index for index, atom_id in enumerate(self.atom_ids_list)
        }
        self._elements_by_base_atom_type = _elements_by_atom_type(
            self.base_atom_types,
            self.elements,
        )
        self.cluster_ids: set[str] = set()
        self.reserved_ids: set[str] = set()
        self._reset_state()

    def _reset_state(self) -> None:
        self.atom_types = dict(self.base_atom_types)
        self.found_status = {atom_id: False for atom_id in self.atom_ids_list}
        self.shell_levels = {atom_id: None for atom_id in self.atom_ids_list}
        self.clusters: list[set[int]] = []
        self.cluster_labels: list[str] = []
        self.atom_virtual_positions: dict[int, np.ndarray] = {}

        for atom in self.atom_id_map.values():
            atom.cluster_id = None
            atom.shell_level = "free" if atom.atom_type == "shell" else None
            atom.shell_history.clear()

    def _pair_cutoff(
        self,
        first_element: str,
        second_element: str,
        shell_level: int,
    ) -> float | None:
        pair = (first_element.title(), second_element.title())
        reverse_pair = (pair[1], pair[0])

        levels = self.pair_cutoffs_def.get(pair)
        if levels is not None and shell_level in levels:
            return levels[shell_level]

        reverse_levels = self.pair_cutoffs_def.get(reverse_pair)
        if reverse_levels is not None and shell_level in reverse_levels:
            return reverse_levels[shell_level]

        return self.default_cutoff

    def _max_cutoff_for_allowed_types(
        self,
        parent_element: str,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
    ) -> float | None:
        candidate_elements: set[str] = set()
        for atom_type in allowed_atom_types:
            candidate_elements.update(
                self._elements_by_base_atom_type.get(atom_type, set())
            )

        max_cutoff: float | None = None
        for candidate_element in candidate_elements:
            cutoff = self._pair_cutoff(
                parent_element,
                candidate_element,
                shell_level,
            )
            if cutoff is None or cutoff <= 0.0:
                continue
            if max_cutoff is None or cutoff > max_cutoff:
                max_cutoff = cutoff
        return max_cutoff

    def _minimum_image_delta(
        self,
        reference_position: np.ndarray,
        other_position: np.ndarray,
    ) -> np.ndarray:
        delta = other_position - reference_position
        if self.use_pbc and self.box is not None:
            positive_axes = self.box > 0.0
            if np.any(positive_axes):
                delta = delta.copy()
                delta[positive_axes] -= (
                    np.round(delta[positive_axes] / self.box[positive_axes])
                    * self.box[positive_axes]
                )
        return delta

    def search_atoms_within_pair_cutoff(
        self,
        parent_id: int,
        parent_virtual_pos: np.ndarray,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
        allow_reuse: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        """Return neighbor atoms matched by the cutoff table."""
        if self.search_mode == SEARCH_MODE_KDTREE:
            return self._search_atoms_within_pair_cutoff_kdtree(
                parent_id,
                parent_virtual_pos,
                shell_level,
                allowed_atom_types=allowed_atom_types,
                allow_reuse=allow_reuse,
            )
        return self._search_atoms_within_pair_cutoff_bruteforce(
            parent_id,
            parent_virtual_pos,
            shell_level,
            allowed_atom_types=allowed_atom_types,
            allow_reuse=allow_reuse,
        )

    def _search_atoms_within_pair_cutoff_bruteforce(
        self,
        parent_id: int,
        parent_virtual_pos: np.ndarray,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
        allow_reuse: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        parent_index = self.atom_index_map[parent_id]
        parent_real_pos = self.original_coordinates[parent_index]
        parent_element = self.elements[parent_id]
        results: list[tuple[int, np.ndarray]] = []

        for neighbor_id in self.atom_ids_list:
            if neighbor_id == parent_id:
                continue
            if self.atom_types[neighbor_id] not in allowed_atom_types:
                continue
            if not allow_reuse and self.found_status[neighbor_id]:
                continue

            cutoff = self._pair_cutoff(
                parent_element,
                self.elements[neighbor_id],
                shell_level,
            )
            if cutoff is None or cutoff <= 0.0:
                continue

            neighbor_real_pos = self.original_coordinates[
                self.atom_index_map[neighbor_id]
            ]
            delta = self._minimum_image_delta(
                parent_real_pos, neighbor_real_pos
            )
            if float(np.linalg.norm(delta)) > cutoff:
                continue
            results.append((neighbor_id, parent_virtual_pos + delta))

        return results

    def _search_atoms_within_pair_cutoff_kdtree(
        self,
        parent_id: int,
        parent_virtual_pos: np.ndarray,
        shell_level: int,
        *,
        allowed_atom_types: set[str],
        allow_reuse: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        if self.kd_tree is None:
            return []

        parent_index = self.atom_index_map[parent_id]
        parent_real_pos = self.original_coordinates[parent_index]
        parent_element = self.elements[parent_id]
        max_cutoff = self._max_cutoff_for_allowed_types(
            parent_element,
            shell_level,
            allowed_atom_types=allowed_atom_types,
        )
        if max_cutoff is None or max_cutoff <= 0.0:
            return []

        candidate_indices: set[int] = set()
        for query_position in _query_positions_for_cutoff(
            parent_real_pos,
            max_cutoff,
            box=self.box,
            use_pbc=self.use_pbc,
        ):
            candidate_indices.update(
                int(index)
                for index in self.kd_tree.query_ball_point(
                    query_position,
                    r=max_cutoff,
                )
            )

        results: list[tuple[int, np.ndarray]] = []
        for neighbor_index in candidate_indices:
            neighbor_id = self.atom_ids_list[neighbor_index]
            if neighbor_id == parent_id:
                continue
            if self.atom_types[neighbor_id] not in allowed_atom_types:
                continue
            if not allow_reuse and self.found_status[neighbor_id]:
                continue

            cutoff = self._pair_cutoff(
                parent_element,
                self.elements[neighbor_id],
                shell_level,
            )
            if cutoff is None or cutoff <= 0.0:
                continue

            neighbor_real_pos = self.original_coordinates[neighbor_index]
            delta = self._minimum_image_delta(
                parent_real_pos,
                neighbor_real_pos,
            )
            if float(np.linalg.norm(delta)) > cutoff:
                continue
            results.append((neighbor_id, parent_virtual_pos + delta))

        return results

    def generate_cluster_id(self) -> str:
        """Generate a unique three-letter cluster identifier."""
        max_ids = 26**3
        for value in range(max_ids):
            temp = value
            cluster_id = ""
            for _ in range(3):
                cluster_id = ascii_uppercase[temp % 26] + cluster_id
                temp //= 26
            if (
                cluster_id not in self.cluster_ids
                and cluster_id not in self.reserved_ids
            ):
                self.cluster_ids.add(cluster_id)
                return cluster_id
        raise ValueError("All possible cluster IDs are reserved or in use.")

    def assign_cluster_id_and_shell_level(
        self,
        cluster: set[int],
        cluster_id: str,
    ) -> None:
        """Apply cluster metadata to the atoms contained in one
        cluster."""
        for atom_id in cluster:
            atom = self.atom_id_map[atom_id]
            atom.update_cluster_assignment(
                cluster_id,
                shell_level=self.shell_levels[atom_id],
            )

    def create_cluster(self, start_atom_id: int) -> set[int]:
        """Build a node-linker cluster starting from one atom."""
        current_cluster = {start_atom_id}
        self.found_status[start_atom_id] = True
        self.shell_levels[start_atom_id] = 0
        start_position = self.original_coordinates[
            self.atom_index_map[start_atom_id]
        ]
        self.atom_virtual_positions[start_atom_id] = start_position.copy()
        search_stack: deque[tuple[int, np.ndarray]] = deque(
            [(start_atom_id, start_position.copy())]
        )

        while search_stack:
            current_id, current_virt_pos = search_stack.popleft()
            for (
                neighbor_id,
                neighbor_virtual_pos,
            ) in self.search_atoms_within_pair_cutoff(
                parent_id=current_id,
                parent_virtual_pos=current_virt_pos,
                shell_level=0,
                allowed_atom_types={"node", "linker", "shell"},
            ):
                if self.found_status[neighbor_id]:
                    continue
                self.found_status[neighbor_id] = True
                self.shell_levels[neighbor_id] = 0
                self.atom_virtual_positions[neighbor_id] = neighbor_virtual_pos
                current_cluster.add(neighbor_id)
                if self.base_atom_types[neighbor_id] != "shell":
                    search_stack.append((neighbor_id, neighbor_virtual_pos))

        return current_cluster

    def process_shell_level_for_cluster(
        self,
        cluster: set[int],
        cluster_id: str,
        shell_level: int,
        *,
        shared_shells: bool = False,
    ) -> None:
        """Add shell atoms at a given shell level to one cluster."""
        node_ids = [
            atom_id
            for atom_id in cluster
            if self.base_atom_types.get(atom_id) == "node"
        ]
        new_shells: dict[int, np.ndarray] = {}

        for node_id in node_ids:
            parent_virtual_pos = self.atom_virtual_positions[node_id]
            for (
                neighbor_id,
                neighbor_virtual_pos,
            ) in self.search_atoms_within_pair_cutoff(
                parent_id=node_id,
                parent_virtual_pos=parent_virtual_pos,
                shell_level=shell_level,
                allowed_atom_types={"shell"},
                allow_reuse=shared_shells,
            ):
                if neighbor_id not in new_shells:
                    new_shells[neighbor_id] = neighbor_virtual_pos

        for neighbor_id, neighbor_virtual_pos in new_shells.items():
            if neighbor_id in cluster:
                continue
            if not shared_shells and self.found_status[neighbor_id]:
                continue
            if not shared_shells:
                self.found_status[neighbor_id] = True
            cluster.add(neighbor_id)
            self.atom_types[neighbor_id] = "shell"
            self.shell_levels[neighbor_id] = shell_level
            self.atom_virtual_positions[neighbor_id] = neighbor_virtual_pos
            self.atom_id_map[neighbor_id].update_cluster_assignment(
                cluster_id,
                shell_level=shell_level,
            )

    def find_clusters(
        self,
        *,
        shell_levels: Sequence[int] = (1, 2),
        shared_shells: bool = False,
    ) -> list[ClusterRecord]:
        """Find cluster networks for the current structure."""
        self._reset_state()

        for atom_id, atom_type in self.base_atom_types.items():
            if atom_type == "node" and not self.found_status[atom_id]:
                cluster = self.create_cluster(atom_id)
                cluster_id = self.generate_cluster_id()
                self.assign_cluster_id_and_shell_level(cluster, cluster_id)
                self.clusters.append(cluster)
                self.cluster_labels.append(cluster_id)

        for atom_id, atom_type in self.base_atom_types.items():
            if atom_type == "linker" and not self.found_status[atom_id]:
                cluster = self.create_cluster(atom_id)
                cluster_id = self.generate_cluster_id()
                self.assign_cluster_id_and_shell_level(cluster, cluster_id)
                self.clusters.append(cluster)
                self.cluster_labels.append(cluster_id)

        for shell_level in shell_levels:
            if shell_level <= 0:
                continue
            for cluster, cluster_id in zip(self.clusters, self.cluster_labels):
                self.process_shell_level_for_cluster(
                    cluster,
                    cluster_id,
                    shell_level,
                    shared_shells=shared_shells,
                )

        return self.cluster_records()

    def build_cluster_atoms(
        self,
        cluster_index: int,
        *,
        include_shell_levels: Sequence[int] = (0, 1, 2),
    ) -> list[XYZAtom]:
        """Build output atoms for one cluster using virtual
        coordinates."""
        cluster = self.clusters[cluster_index]
        atom_ids = {
            atom_id
            for atom_id in cluster
            if self.shell_levels.get(atom_id) in include_shell_levels
        }
        if not atom_ids:
            return []

        final_atoms: list[XYZAtom] = []
        for atom_id in sorted(atom_ids):
            atom = self.all_atoms_map.get(atom_id)
            if atom is None:
                continue
            copied_atom = atom.copy()
            virtual_position = self.atom_virtual_positions.get(atom_id)
            if virtual_position is not None:
                copied_atom.coordinates = virtual_position.copy()
            final_atoms.append(copied_atom)

        return final_atoms

    def write_cluster_xyz_files(
        self,
        output_dir: str | Path,
        *,
        frame_label: str | None = None,
        include_shell_levels: Sequence[int] = (0, 1, 2),
    ) -> list[Path]:
        """Write per-cluster XYZ files for the current structure."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written_files: list[Path] = []
        source_label = (
            frame_label or self.xyz_structure.source_name or "cluster_frame"
        )

        for cluster_index, cluster_id in enumerate(self.cluster_labels):
            atoms = self.build_cluster_atoms(
                cluster_index,
                include_shell_levels=include_shell_levels,
            )
            if not atoms:
                continue
            output_path = output_dir / f"{source_label}_{cluster_id}.xyz"
            self.xyz_structure.write_xyz_file(
                output_path,
                atoms,
                comment=(
                    f"frame={source_label} cluster={cluster_id} "
                    "generated by SAXShell cluster analysis"
                ),
            )
            written_files.append(output_path)

        return written_files

    def cluster_records(self) -> list[ClusterRecord]:
        """Return cluster summary records for the current structure."""
        records: list[ClusterRecord] = []
        for cluster, cluster_id in zip(self.clusters, self.cluster_labels):
            atom_ids = tuple(sorted(cluster))
            node_atom_ids = tuple(
                atom_id
                for atom_id in atom_ids
                if self.base_atom_types.get(atom_id) == "node"
            )
            linker_atom_ids = tuple(
                atom_id
                for atom_id in atom_ids
                if self.base_atom_types.get(atom_id) == "linker"
            )
            shell_atom_ids = tuple(
                atom_id
                for atom_id in atom_ids
                if self.atom_types.get(atom_id) == "shell"
                and atom_id not in node_atom_ids
                and atom_id not in linker_atom_ids
            )
            solute_atom_ids = tuple(sorted(node_atom_ids + linker_atom_ids))
            stoichiometry_atom_ids = list(solute_atom_ids)
            if self.include_shell_atoms_in_stoichiometry:
                stoichiometry_atom_ids.extend(shell_atom_ids)
            stoichiometry = dict(
                Counter(
                    self.elements[atom_id]
                    for atom_id in stoichiometry_atom_ids
                )
            )
            shell_level_map = {
                atom_id: self.shell_levels.get(atom_id) for atom_id in atom_ids
            }
            records.append(
                ClusterRecord(
                    cluster_id=cluster_id,
                    atom_ids=atom_ids,
                    solute_atom_ids=solute_atom_ids,
                    node_atom_ids=node_atom_ids,
                    linker_atom_ids=linker_atom_ids,
                    shell_atom_ids=shell_atom_ids,
                    stoichiometry=stoichiometry,
                    shell_levels=shell_level_map,
                )
            )
        return records


class TrajectoryClusterAnalyzer:
    """Run :class:`ClusterNetwork` across frames of a PDB trajectory."""

    def __init__(
        self,
        trajectory_file: str | Path,
        atom_type_definitions: AtomTypeDefinitions,
        pair_cutoffs_def: Mapping[
            tuple[str, str],
            Mapping[int, float] | float | int,
        ],
        *,
        topology_file: str | Path | None = None,
        box_dimensions: Sequence[float] | None = None,
        default_cutoff: float | None = None,
        use_pbc: bool = False,
        include_shell_atoms_in_stoichiometry: bool = False,
        search_mode: str = SEARCH_MODE_KDTREE,
        backend: str = "auto",
    ) -> None:
        self.trajectory_file = Path(trajectory_file)
        self.atom_type_definitions = atom_type_definitions
        self.pair_cutoffs_def = pair_cutoffs_def
        self.box_dimensions = box_dimensions
        self.default_cutoff = default_cutoff
        self.use_pbc = bool(use_pbc)
        self.include_shell_atoms_in_stoichiometry = bool(
            include_shell_atoms_in_stoichiometry
        )
        self.search_mode = normalize_search_mode(search_mode)
        self.manager = TrajectoryManager(
            input_file=self.trajectory_file,
            topology_file=topology_file,
            backend=backend,
        )
        self.last_results: list[FrameClusterResult] = []

    def inspect(self) -> dict[str, object]:
        """Inspect the underlying trajectory and validate its type."""
        summary = self.manager.inspect()
        if summary.get("file_type") != "pdb":
            raise ValueError(
                "Cluster analysis currently requires a PDB trajectory."
            )
        return summary

    def analyze_frames(
        self,
        *,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        min_time_fs: float | None = None,
        shell_levels: Sequence[int] = (1, 2),
        shared_shells: bool = False,
    ) -> list[FrameClusterResult]:
        """Analyze each selected trajectory frame."""
        self.inspect()
        frames = self.manager.get_selected_frames(
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=min_time_fs,
        )

        results: list[FrameClusterResult] = []
        for frame in frames:
            structure = PDBStructure.from_lines(
                frame.lines,
                atom_type_definitions=self.atom_type_definitions,
                source_name=f"frame_{frame.frame_index:04d}",
            )
            network = ClusterNetwork(
                pdb_structure=structure,
                atom_type_definitions=self.atom_type_definitions,
                pair_cutoffs_def=self.pair_cutoffs_def,
                box_dimensions=self.box_dimensions,
                default_cutoff=self.default_cutoff,
                use_pbc=self.use_pbc,
                include_shell_atoms_in_stoichiometry=(
                    self.include_shell_atoms_in_stoichiometry
                ),
                search_mode=self.search_mode,
            )
            clusters = network.find_clusters(
                shell_levels=shell_levels,
                shared_shells=shared_shells,
            )
            results.append(
                FrameClusterResult(
                    frame_index=frame.frame_index,
                    time_fs=frame.time_fs,
                    clusters=clusters,
                )
            )

        self.last_results = results
        return results

    def export_cluster_pdbs(
        self,
        output_dir: str | Path,
        *,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        min_time_fs: float | None = None,
        shell_levels: Sequence[int] = (1, 2),
        include_shell_levels: Sequence[int] = (0, 1, 2),
        shared_shells: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> TrajectoryClusterExport:
        """Analyze frames and write per-cluster PDB files grouped by
        stoichiometry."""
        self.inspect()
        frames = self.manager.get_selected_frames(
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=min_time_fs,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written_files: list[Path] = []
        frame_results: list[FrameClusterResult] = []
        total_frames = len(frames)
        pending_sort_jobs: list[tuple[Path, str, list[ClusterRecord]]] = []
        stoichiometry_dir_cache: dict[str, Path] = {}

        for processed_count, frame in enumerate(frames, start=1):
            structure = PDBStructure.from_lines(
                frame.lines,
                atom_type_definitions=self.atom_type_definitions,
                source_name=f"frame_{frame.frame_index:04d}",
            )
            network = ClusterNetwork(
                pdb_structure=structure,
                atom_type_definitions=self.atom_type_definitions,
                pair_cutoffs_def=self.pair_cutoffs_def,
                box_dimensions=self.box_dimensions,
                default_cutoff=self.default_cutoff,
                use_pbc=self.use_pbc,
                include_shell_atoms_in_stoichiometry=(
                    self.include_shell_atoms_in_stoichiometry
                ),
                search_mode=self.search_mode,
            )
            clusters = network.find_clusters(
                shell_levels=shell_levels,
                shared_shells=shared_shells,
            )
            frame_results.append(
                FrameClusterResult(
                    frame_index=frame.frame_index,
                    time_fs=frame.time_fs,
                    clusters=clusters,
                )
            )
            frame_label = f"frame_{frame.frame_index:04d}"
            frame_dir = output_dir / f"frame_{frame.frame_index:04d}"
            network.write_cluster_pdb_files(
                frame_dir,
                frame_label=frame_label,
                include_shell_levels=include_shell_levels,
            )
            pending_sort_jobs.append((frame_dir, frame_label, clusters))
            if progress_callback is not None:
                progress_callback(processed_count, total_frames, frame_label)

        for frame_dir, frame_label, clusters in pending_sort_jobs:
            written_files.extend(
                _move_cluster_files_to_stoichiometry_dirs(
                    output_dir,
                    frame_dir,
                    frame_label,
                    clusters,
                    suffix=".pdb",
                    stoichiometry_dir_cache=stoichiometry_dir_cache,
                )
            )

        self.last_results = frame_results
        return TrajectoryClusterExport(
            written_files=written_files,
            frame_results=frame_results,
        )


class ExtractedFrameFolderClusterAnalyzer:
    """Run cluster analysis across extracted single-frame PDB or XYZ
    files."""

    def __init__(
        self,
        frames_dir: str | Path,
        atom_type_definitions: AtomTypeDefinitions,
        pair_cutoffs_def: Mapping[
            tuple[str, str],
            Mapping[int, float] | float | int,
        ],
        *,
        box_dimensions: Sequence[float] | None = None,
        default_cutoff: float | None = None,
        use_pbc: bool = False,
        include_shell_atoms_in_stoichiometry: bool = False,
        search_mode: str = SEARCH_MODE_KDTREE,
        save_state_frequency: int = DEFAULT_SAVE_STATE_FREQUENCY,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.atom_type_definitions = atom_type_definitions
        self.pair_cutoffs_def = pair_cutoffs_def
        self.box_dimensions = box_dimensions
        self.default_cutoff = default_cutoff
        self.use_pbc = bool(use_pbc)
        self.include_shell_atoms_in_stoichiometry = bool(
            include_shell_atoms_in_stoichiometry
        )
        self.search_mode = normalize_search_mode(search_mode)
        self.save_state_frequency = normalize_save_state_frequency(
            save_state_frequency
        )
        self.last_results: list[FrameClusterResult] = []
        self._cached_frame_format: str | None = None
        self._cached_frame_paths: list[Path] | None = None
        self._cached_summary: dict[str, object] | None = None

    def inspect(self) -> dict[str, object]:
        """Inspect the input folder and summarize extracted frame
        files."""
        if self._cached_summary is not None:
            return dict(self._cached_summary)

        frame_format, frame_paths = self._detect_frames()
        box_dimensions: tuple[float, float, float] | None = None
        box_source: str | None = None
        box_source_kind: str | None = None

        if frame_format == "xyz":
            detected_source = detect_source_box_dimensions(self.frames_dir)
            if detected_source is not None:
                box_dimensions, source_path = detected_source
                box_source = source_path.name
                box_source_kind = "source_filename"

        if box_dimensions is None:
            box_dimensions = self._estimate_box_dimensions(
                frame_format,
                frame_paths[0],
            )
            if box_dimensions is not None:
                box_source = frame_paths[0].name
                box_source_kind = "estimate"

        self._cached_summary = {
            "input_dir": str(self.frames_dir),
            "file_type": f"{frame_format}_frames",
            "frame_format": frame_format,
            "mode_label": frame_folder_label(frame_format),
            "output_file_extension": frame_output_suffix(frame_format),
            "supports_full_molecule_shells": frame_format == "pdb",
            "estimated_box_dimensions": box_dimensions,
            "box_dimensions": box_dimensions,
            "box_dimensions_source": box_source,
            "box_dimensions_source_kind": box_source_kind,
            "box_estimate_source": (
                box_source if box_source_kind == "estimate" else None
            ),
            "n_frames": len(frame_paths),
            "first_frame": frame_paths[0].name,
            "last_frame": frame_paths[-1].name,
        }
        return dict(self._cached_summary)

    def analyze_frames(
        self,
        *,
        shell_levels: Sequence[int] = (1, 2),
        shared_shells: bool = False,
    ) -> list[FrameClusterResult]:
        """Analyze each extracted frame in the selected folder."""
        _frame_format, frame_paths = self._validated_frame_paths()
        results: list[FrameClusterResult] = []

        for frame_index, frame_path in enumerate(frame_paths):
            network = self._build_network(frame_path)
            clusters = network.find_clusters(
                shell_levels=shell_levels,
                shared_shells=shared_shells,
            )
            results.append(
                FrameClusterResult(
                    frame_index=frame_index,
                    time_fs=None,
                    clusters=clusters,
                )
            )

        self.last_results = results
        return results

    def export_cluster_files(
        self,
        output_dir: str | Path,
        *,
        shell_levels: Sequence[int] = (1, 2),
        include_shell_levels: Sequence[int] = (0, 1, 2),
        shared_shells: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
        phase_callback: Callable[[str, int, int], None] | None = None,
    ) -> TrajectoryClusterExport:
        """Analyze extracted frames and write per-cluster output
        files."""
        frame_format, frame_paths = self._validated_frame_paths()
        summary = self.inspect()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = _metadata_file_path(output_dir)
        total_frames = len(frame_paths)

        metadata = self._load_or_initialize_metadata(
            metadata_path=metadata_path,
            output_dir=output_dir,
            summary=summary,
            frame_format=frame_format,
            frame_paths=frame_paths,
            shell_levels=shell_levels,
            include_shell_levels=include_shell_levels,
            shared_shells=shared_shells,
        )
        frame_entries = {
            str(entry["frame_name"]): dict(entry)
            for entry in metadata.get("frames", [])
        }
        previously_processed_frames = sum(
            1
            for entry in frame_entries.values()
            if _frame_entry_is_processed(entry)
        )
        previously_completed_frames = sum(
            1
            for entry in frame_entries.values()
            if _frame_entry_is_completed(entry)
        )
        resumed = previously_processed_frames > 0

        if progress_callback is not None:
            progress_callback(
                previously_processed_frames,
                total_frames,
                "resume",
            )
        if phase_callback is not None:
            phase_callback(
                "extracting",
                previously_processed_frames,
                total_frames,
            )

        if previously_completed_frames >= total_frames:
            metadata["state"] = "completed"
            metadata["updated_at"] = _utc_timestamp()
            metadata["last_error"] = None
            self._refresh_metadata_progress(
                metadata, frame_entries, total_frames
            )
            self._store_metadata(metadata_path, metadata, frame_entries)
            frame_results = self._frame_results_from_entries(frame_entries)
            written_files = self._written_files_from_metadata(
                output_dir,
                metadata,
                frame_entries,
            )
            self.last_results = frame_results
            return TrajectoryClusterExport(
                written_files=written_files,
                frame_results=frame_results,
                metadata_path=metadata_path,
                resumed=resumed,
                already_complete=True,
                previously_completed_frames=previously_processed_frames,
                newly_processed_frames=0,
            )

        metadata["state"] = "in_progress"
        metadata["updated_at"] = _utc_timestamp()
        metadata["completed_at"] = None
        metadata["last_error"] = None
        self._refresh_metadata_progress(
            metadata,
            frame_entries,
            total_frames,
        )
        self._store_metadata(metadata_path, metadata, frame_entries)

        newly_processed_frames = 0
        frames_since_checkpoint = 0
        last_checkpoint_time = monotonic()
        sorting_completed_frames = previously_completed_frames
        stoichiometry_dir_cache: dict[str, Path] = {}

        def checkpoint_metadata(*, force: bool = False) -> None:
            nonlocal frames_since_checkpoint, last_checkpoint_time
            if not force:
                if (
                    frames_since_checkpoint < self.save_state_frequency
                    and (monotonic() - last_checkpoint_time)
                    < METADATA_CHECKPOINT_SECONDS
                ):
                    return
            metadata["updated_at"] = _utc_timestamp()
            self._store_metadata(metadata_path, metadata, frame_entries)
            frames_since_checkpoint = 0
            last_checkpoint_time = monotonic()

        try:
            for frame_index, frame_path in enumerate(frame_paths):
                existing_entry = frame_entries.get(frame_path.name)
                if existing_entry is not None and _frame_entry_is_processed(
                    existing_entry
                ):
                    continue

                network = self._build_network(frame_path)
                clusters = network.find_clusters(
                    shell_levels=shell_levels,
                    shared_shells=shared_shells,
                )
                frame_result = FrameClusterResult(
                    frame_index=frame_index,
                    time_fs=None,
                    clusters=clusters,
                )
                frame_label = frame_path.stem
                frame_dir = output_dir / frame_path.stem
                if frame_format == "pdb":
                    temporary_files = network.write_cluster_pdb_files(
                        frame_dir,
                        frame_label=frame_label,
                        include_shell_levels=include_shell_levels,
                    )
                else:
                    temporary_files = network.write_cluster_xyz_files(
                        frame_dir,
                        frame_label=frame_label,
                        include_shell_levels=include_shell_levels,
                    )
                frame_entries[frame_path.name] = {
                    "frame_name": frame_path.name,
                    "frame_label": frame_label,
                    "frame_index": frame_index,
                    "status": FRAME_STATUS_PROCESSED,
                    "processed_at": _utc_timestamp(),
                    "completed_at": None,
                    "temporary_files": _relative_output_paths(
                        output_dir,
                        temporary_files,
                    ),
                    "written_files": [],
                    "result": frame_result.to_dict(),
                }
                newly_processed_frames += 1
                metadata["last_error"] = None
                frames_since_checkpoint += 1
                checkpoint_metadata()

                if progress_callback is not None:
                    progress_callback(
                        previously_processed_frames + newly_processed_frames,
                        total_frames,
                        frame_label,
                    )

            metadata["state"] = "sorting"
            checkpoint_metadata(force=True)
            if phase_callback is not None:
                phase_callback(
                    "sorting",
                    sorting_completed_frames,
                    total_frames,
                )

            for frame_index, frame_path in enumerate(frame_paths):
                entry = frame_entries.get(frame_path.name)
                if entry is None or _frame_entry_is_completed(entry):
                    continue

                frame_label = str(entry.get("frame_label", frame_path.stem))
                frame_result = FrameClusterResult.from_dict(
                    dict(entry["result"])
                )
                frame_dir = output_dir / frame_label
                moved_files = _move_cluster_files_to_stoichiometry_dirs(
                    output_dir,
                    frame_dir,
                    frame_label,
                    frame_result.clusters,
                    suffix=frame_output_suffix(frame_format),
                    stoichiometry_dir_cache=stoichiometry_dir_cache,
                )
                entry.update(
                    {
                        "frame_name": frame_path.name,
                        "frame_label": frame_label,
                        "frame_index": frame_index,
                        "status": FRAME_STATUS_COMPLETED,
                        "processed_at": entry.get("processed_at"),
                        "completed_at": _utc_timestamp(),
                        "temporary_files": [],
                        "written_files": _relative_output_paths(
                            output_dir,
                            moved_files,
                        ),
                        "result": frame_result.to_dict(),
                    }
                )
                sorting_completed_frames += 1
                frames_since_checkpoint += 1
                checkpoint_metadata()
                if progress_callback is not None:
                    progress_callback(
                        sorting_completed_frames,
                        total_frames,
                        frame_label,
                    )
        except Exception as exc:
            metadata["state"] = "failed"
            metadata["updated_at"] = _utc_timestamp()
            metadata["last_error"] = str(exc)
            checkpoint_metadata(force=True)
            raise

        metadata["state"] = "completed"
        metadata["updated_at"] = _utc_timestamp()
        metadata["completed_at"] = _utc_timestamp()
        metadata["last_error"] = None
        checkpoint_metadata(force=True)

        frame_results = self._frame_results_from_entries(frame_entries)
        written_files = self._written_files_from_metadata(
            output_dir,
            metadata,
            frame_entries,
        )
        self.last_results = frame_results
        return TrajectoryClusterExport(
            written_files=written_files,
            frame_results=frame_results,
            metadata_path=metadata_path,
            resumed=resumed,
            already_complete=False,
            previously_completed_frames=previously_processed_frames,
            newly_processed_frames=newly_processed_frames,
        )

    def export_cluster_pdbs(
        self,
        output_dir: str | Path,
        *,
        shell_levels: Sequence[int] = (1, 2),
        include_shell_levels: Sequence[int] = (0, 1, 2),
        shared_shells: bool = False,
    ) -> TrajectoryClusterExport:
        """Compatibility wrapper for the older export method name."""
        return self.export_cluster_files(
            output_dir,
            shell_levels=shell_levels,
            include_shell_levels=include_shell_levels,
            shared_shells=shared_shells,
        )

    def _validated_frame_paths(self) -> tuple[str, list[Path]]:
        summary = self.inspect()
        assert summary["n_frames"]
        return str(summary["frame_format"]), self._frame_paths()

    def _frame_paths(self) -> list[Path]:
        _frame_format, frame_paths = self._detect_frames()
        return list(frame_paths)

    def _build_network(
        self,
        frame_path: Path,
    ) -> ClusterNetwork | XYZClusterNetwork:
        frame_format, _frame_paths = self._detect_frames()
        if frame_format == "pdb":
            structure = self._load_structure(
                frame_format,
                frame_path,
                atom_type_definitions=self.atom_type_definitions,
            )
            return ClusterNetwork(
                pdb_structure=structure,
                atom_type_definitions=self.atom_type_definitions,
                pair_cutoffs_def=self.pair_cutoffs_def,
                box_dimensions=self.box_dimensions,
                default_cutoff=self.default_cutoff,
                use_pbc=self.use_pbc,
                include_shell_atoms_in_stoichiometry=(
                    self.include_shell_atoms_in_stoichiometry
                ),
                search_mode=self.search_mode,
            )

        structure = self._load_structure(
            frame_format,
            frame_path,
            atom_type_definitions=self.atom_type_definitions,
        )
        return XYZClusterNetwork(
            xyz_structure=structure,
            atom_type_definitions=self.atom_type_definitions,
            pair_cutoffs_def=self.pair_cutoffs_def,
            box_dimensions=self.box_dimensions,
            default_cutoff=self.default_cutoff,
            use_pbc=self.use_pbc,
            include_shell_atoms_in_stoichiometry=(
                self.include_shell_atoms_in_stoichiometry
            ),
            search_mode=self.search_mode,
        )

    def _detect_frames(self) -> tuple[str, list[Path]]:
        if (
            self._cached_frame_format is None
            or self._cached_frame_paths is None
        ):
            frame_format, frame_paths = detect_frame_folder_mode(
                self.frames_dir
            )
            self._cached_frame_format = str(frame_format)
            self._cached_frame_paths = list(frame_paths)
        return self._cached_frame_format, list(self._cached_frame_paths)

    def _load_or_initialize_metadata(
        self,
        *,
        metadata_path: Path,
        output_dir: Path,
        summary: Mapping[str, object],
        frame_format: str,
        frame_paths: Sequence[Path],
        shell_levels: Sequence[int],
        include_shell_levels: Sequence[int],
        shared_shells: bool,
    ) -> dict[str, object]:
        signature = self._metadata_signature(
            output_dir=output_dir,
            summary=summary,
            frame_format=frame_format,
            frame_paths=frame_paths,
            shell_levels=shell_levels,
            include_shell_levels=include_shell_levels,
            shared_shells=shared_shells,
        )
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") != signature:
                raise ValueError(
                    "The selected output directory already contains cluster "
                    "metadata for a different extraction setup. Choose a "
                    "new output directory or remove the existing metadata "
                    "file before exporting."
                )
            self._update_runtime_metadata(metadata)
            return metadata

        timestamp = _utc_timestamp()
        metadata = {
            "format_version": 1,
            "metadata_file": metadata_path.name,
            "state": "pending",
            "created_at": timestamp,
            "updated_at": timestamp,
            "completed_at": None,
            "last_error": None,
            "signature": signature,
            "input": {
                "frames_dir": str(self.frames_dir.resolve()),
                "frame_format": frame_format,
                "n_frames": int(summary["n_frames"]),
                "first_frame": summary.get("first_frame"),
                "last_frame": summary.get("last_frame"),
                "frame_manifest": [
                    _frame_manifest_entry(frame_path)
                    for frame_path in frame_paths
                ],
            },
            "parameters": signature["parameters"],
            "output": {
                "output_dir": str(output_dir.resolve()),
                "written_files": [],
            },
            "progress": {
                "total_frames": len(frame_paths),
                "completed_frames": 0,
                "remaining_frames": len(frame_paths),
                "completed_frame_names": [],
            },
            "frames": [],
        }
        self._update_runtime_metadata(metadata)
        return metadata

    def _metadata_signature(
        self,
        *,
        output_dir: Path,
        summary: Mapping[str, object],
        frame_format: str,
        frame_paths: Sequence[Path],
        shell_levels: Sequence[int],
        include_shell_levels: Sequence[int],
        shared_shells: bool,
    ) -> dict[str, object]:
        return {
            "frames_dir": str(self.frames_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "frame_format": frame_format,
            "frame_manifest": [
                _frame_manifest_entry(frame_path) for frame_path in frame_paths
            ],
            "parameters": {
                "atom_type_definitions": _serialize_atom_type_definitions(
                    self.atom_type_definitions
                ),
                "pair_cutoffs_def": _serialize_pair_cutoff_definitions(
                    self.pair_cutoffs_def
                ),
                "box_dimensions": (
                    None
                    if self.box_dimensions is None
                    else [float(value) for value in self.box_dimensions]
                ),
                "default_cutoff": self.default_cutoff,
                "use_pbc": self.use_pbc,
                "include_shell_atoms_in_stoichiometry": (
                    self.include_shell_atoms_in_stoichiometry
                ),
                "search_mode": self.search_mode,
                "shell_levels": [int(level) for level in shell_levels],
                "include_shell_levels": [
                    int(level) for level in include_shell_levels
                ],
                "shared_shells": bool(shared_shells),
                "first_frame": summary.get("first_frame"),
                "last_frame": summary.get("last_frame"),
            },
        }

    def _refresh_metadata_progress(
        self,
        metadata: dict[str, object],
        frame_entries: Mapping[str, Mapping[str, object]],
        total_frames: int,
    ) -> None:
        processed_entries = [
            entry
            for entry in frame_entries.values()
            if _frame_entry_is_processed(entry)
        ]
        completed_entries = [
            entry
            for entry in frame_entries.values()
            if _frame_entry_is_completed(entry)
        ]
        processed_names = sorted(
            str(entry["frame_name"]) for entry in processed_entries
        )
        completed_names = sorted(
            str(entry["frame_name"]) for entry in completed_entries
        )
        metadata["frames"] = [
            dict(entry)
            for entry in sorted(
                frame_entries.values(),
                key=_entry_sort_key,
            )
        ]
        metadata["progress"] = {
            "total_frames": total_frames,
            "processed_frames": len(processed_entries),
            "completed_frames": len(processed_entries),
            "sorted_frames": len(completed_entries),
            "remaining_frames": max(total_frames - len(processed_entries), 0),
            "sorting_remaining_frames": max(
                len(processed_entries) - len(completed_entries),
                0,
            ),
            "processed_frame_names": processed_names,
            "completed_frame_names": processed_names,
            "sorted_frame_names": completed_names,
        }
        written_files = sorted(
            {
                str(relative_path)
                for entry in completed_entries
                for relative_path in entry.get("written_files", [])
            }
        )
        metadata["output"] = {
            "output_dir": str(
                Path(str(metadata["output"]["output_dir"])).resolve()
            ),
            "written_files": written_files,
        }

    def _update_runtime_metadata(self, metadata: dict[str, object]) -> None:
        """Refresh runtime-only metadata fields for the current run."""
        metadata["runtime"] = {
            "save_state_frequency_frames": self.save_state_frequency,
            "checkpoint_seconds": METADATA_CHECKPOINT_SECONDS,
        }

    def _store_metadata(
        self,
        metadata_path: Path,
        metadata: dict[str, object],
        frame_entries: Mapping[str, Mapping[str, object]],
    ) -> None:
        metadata["updated_at"] = _utc_timestamp()
        self._update_runtime_metadata(metadata)
        if "progress" in metadata:
            total_frames = int(
                dict(metadata["progress"]).get("total_frames", 0)
            )
            self._refresh_metadata_progress(
                metadata,
                frame_entries,
                total_frames,
            )
        _write_json_file(metadata_path, metadata)

    def _frame_results_from_entries(
        self,
        frame_entries: Mapping[str, Mapping[str, object]],
    ) -> list[FrameClusterResult]:
        return [
            FrameClusterResult.from_dict(dict(entry["result"]))
            for entry in sorted(frame_entries.values(), key=_entry_sort_key)
            if "result" in entry
        ]

    def _written_files_from_metadata(
        self,
        output_dir: Path,
        metadata: Mapping[str, object],
        frame_entries: Mapping[str, Mapping[str, object]],
    ) -> list[Path]:
        output_payload = dict(metadata.get("output", {}))
        relative_paths = list(output_payload.get("written_files", []))
        if not relative_paths:
            relative_paths = sorted(
                {
                    str(relative_path)
                    for entry in frame_entries.values()
                    for relative_path in entry.get("written_files", [])
                }
            )
        return _absolute_output_paths(output_dir, relative_paths)

    def _estimate_box_dimensions(
        self,
        frame_format: str,
        frame_path: Path,
    ) -> tuple[float, float, float] | None:
        structure = self._load_structure(
            frame_format,
            frame_path,
            atom_type_definitions={},
        )
        coordinates = [atom.coordinates for atom in structure.atoms]
        return estimate_box_dimensions_from_coordinates(coordinates)

    @staticmethod
    def _load_structure(
        frame_format: str,
        frame_path: Path,
        *,
        atom_type_definitions: AtomTypeDefinitions,
    ) -> PDBStructure | XYZStructure:
        if frame_format == "pdb":
            return PDBStructure(
                filepath=frame_path,
                atom_type_definitions=atom_type_definitions,
                source_name=frame_path.stem,
            )
        return XYZStructure(
            filepath=frame_path,
            atom_type_definitions=atom_type_definitions,
            source_name=frame_path.stem,
        )
