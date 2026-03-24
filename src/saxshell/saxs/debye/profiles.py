from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

try:
    import xraydb  # type: ignore
except ImportError:  # pragma: no cover - optional import until runtime
    xraydb = None


@dataclass(slots=True, frozen=True)
class ClusterBin:
    structure: str
    motif: str
    source_dir: Path
    files: tuple[Path, ...]
    representative: str | None


@dataclass(slots=True, frozen=True)
class AveragedComponent:
    structure: str
    motif: str
    file_count: int
    representative: str | None
    source_dir: Path
    q_values: np.ndarray
    mean_intensity: np.ndarray
    std_intensity: np.ndarray
    se_intensity: np.ndarray
    output_path: Path


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
        if token
    ]


@lru_cache(maxsize=None)
def _normalized_element_symbol(raw_value: str) -> str:
    text = raw_value.strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def _require_xraydb() -> None:
    if xraydb is None:
        raise RuntimeError(
            "xraydb is required for SAXS Debye component calculation because "
            "the implementation preserves the legacy f0(q) scattering logic."
        )


def load_structure_file(file_path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return _load_pdb(path)
    if suffix == ".xyz":
        return _load_xyz(path)
    raise ValueError(f"Unsupported structure format: {path.suffix}")


def scan_structure_elements(file_path: str | Path) -> list[str]:
    """Read only the element symbols from a structure file.

    This is intentionally lighter than ``load_structure_file`` so directory
    import/inspection does not spend time parsing coordinates that are not yet
    needed.
    """

    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return _scan_pdb_elements(path)
    if suffix == ".xyz":
        return _scan_xyz_elements(path)
    raise ValueError(f"Unsupported structure format: {path.suffix}")


def scan_structure_element_counts(file_path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in scan_structure_elements(file_path):
        counts[element] = counts.get(element, 0) + 1
    return counts


def _load_pdb(path: Path) -> tuple[np.ndarray, list[str]]:
    positions: list[list[float]] = []
    elements: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                x_coord = float(line[30:38].strip())
                y_coord = float(line[38:46].strip())
                z_coord = float(line[46:54].strip())
                element = line[76:78].strip()
                if not element:
                    element = "".join(
                        char for char in atom_name if char.isalpha()
                    )[:2].strip()
                if not element:
                    warnings.warn(
                        "Unable to determine element for atom "
                        f"{atom_name!r} in {path}. Using 'Unknown'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    element = "Unknown"
                positions.append([x_coord, y_coord, z_coord])
                elements.append(_normalized_element_symbol(element))
            except Exception as exc:
                warnings.warn(
                    f"Skipping malformed PDB atom line {line_number} in {path}: "
                    f"{exc}",
                    UserWarning,
                    stacklevel=2,
                )

    if not positions:
        raise ValueError(f"No atoms were parsed from {path}")
    return np.asarray(positions, dtype=float), elements


def _load_xyz(path: Path) -> tuple[np.ndarray, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file is too short: {path}")
    n_atoms = int(lines[0].strip())
    data_lines = lines[2 : 2 + n_atoms]
    coords: list[list[float]] = []
    elements: list[str] = []
    for line in data_lines:
        tokens = line.split()
        if len(tokens) < 4:
            continue
        elements.append(_normalized_element_symbol(tokens[0]))
        coords.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
    if not coords:
        raise ValueError(f"No atoms were parsed from {path}")
    return np.asarray(coords, dtype=float), elements


def _scan_pdb_elements(path: Path) -> list[str]:
    elements: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip()
            if not element:
                element = "".join(
                    char for char in atom_name if char.isalpha()
                )[:2].strip()
            normalized = _normalized_element_symbol(element)
            if normalized:
                elements.append(normalized)
    if not elements:
        raise ValueError(f"No atoms were parsed from {path}")
    return elements


def _scan_xyz_elements(path: Path) -> list[str]:
    elements: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        atom_count_line = handle.readline()
        if not atom_count_line:
            raise ValueError(f"XYZ file is too short: {path}")
        try:
            atom_count = int(atom_count_line.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ atom count in {path}") from exc
        next(handle, None)
        for atom_index, line in enumerate(handle):
            if atom_index >= atom_count:
                break
            if not line.strip():
                continue
            tokens = line.split(None, 1)
            if not tokens:
                continue
            element = tokens[0]
            normalized = _normalized_element_symbol(element)
            if normalized:
                elements.append(normalized)
    if not elements:
        raise ValueError(f"No atoms were parsed from {path}")
    return elements


def build_f0_dictionary(
    elements: list[str] | tuple[str, ...],
    q_values: np.ndarray,
) -> dict[str, np.ndarray]:
    _require_xraydb()
    normalized_elements = sorted(
        {
            _normalized_element_symbol(element)
            for element in elements
            if _normalized_element_symbol(element)
        }
    )
    f0_dictionary: dict[str, np.ndarray] = {}
    for element in normalized_elements:
        try:
            f0_values = []
            for q_value in q_values:
                sin_theta_over_lambda = float(q_value) / (4.0 * np.pi)
                f0_value = float(xraydb.f0(element, sin_theta_over_lambda)[0])
                f0_values.append(f0_value)
            f0_dictionary[element] = np.asarray(f0_values, dtype=float)
        except Exception as exc:
            warnings.warn(
                f"Failed to compute f0(q) for element {element!r}: {exc}. "
                "Using zeros instead.",
                UserWarning,
                stacklevel=2,
            )
            f0_dictionary[element] = np.zeros_like(q_values, dtype=float)
    return f0_dictionary


def atomic_form_factor(element: str, q_values: np.ndarray) -> np.ndarray:
    symbol = _normalized_element_symbol(element)
    return build_f0_dictionary([symbol], np.asarray(q_values, dtype=float))[
        symbol
    ]


def _filter_atoms_by_elements(
    coordinates: np.ndarray,
    elements: list[str],
    *,
    include_elements: set[str] | None = None,
    exclude_elements: set[str] | None = None,
) -> tuple[np.ndarray, list[str]] | None:
    keep_mask = np.ones(len(elements), dtype=bool)
    if include_elements:
        keep_mask &= np.asarray(
            [element in include_elements for element in elements],
            dtype=bool,
        )
    if exclude_elements:
        keep_mask &= np.asarray(
            [element not in exclude_elements for element in elements],
            dtype=bool,
        )
    if not np.any(keep_mask):
        return None
    filtered_coords = np.asarray(coordinates, dtype=float)[keep_mask]
    filtered_elements = [
        element for element, keep in zip(elements, keep_mask) if keep
    ]
    return filtered_coords, filtered_elements


def compute_debye_intensity(
    coordinates: np.ndarray,
    elements: list[str],
    q_values: np.ndarray,
    *,
    exclude_elements: list[str] | tuple[str, ...] | None = None,
    f0_dictionary: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.size == 0:
        return np.zeros_like(q_values)

    normalized_elements = [
        _normalized_element_symbol(element) for element in elements
    ]
    exclude_set = {
        _normalized_element_symbol(element)
        for element in (exclude_elements or [])
        if _normalized_element_symbol(element)
    }
    filtered = _filter_atoms_by_elements(
        coordinates,
        normalized_elements,
        exclude_elements=exclude_set or None,
    )
    if filtered is None:
        return np.zeros_like(q_values)
    filtered_coords, filtered_elements = filtered

    active_f0_dictionary = (
        f0_dictionary
        if f0_dictionary is not None
        else build_f0_dictionary(filtered_elements, q_values)
    )

    missing = set(filtered_elements) - set(active_f0_dictionary.keys())
    if missing:
        warnings.warn(
            "Missing f0 for elements: "
            + ", ".join(sorted(missing))
            + ". Substituting zeros.",
            UserWarning,
            stacklevel=2,
        )

    f0_matrix = np.vstack(
        [
            active_f0_dictionary.get(
                element,
                np.zeros_like(q_values, dtype=float),
            )
            for element in filtered_elements
        ]
    )
    rij = squareform(pdist(filtered_coords, metric="euclidean"))
    sinc_matrix = np.sinc(rij[:, :, None] * q_values[None, None, :] / np.pi)
    return np.einsum("ijm,im,jm->m", sinc_matrix, f0_matrix, f0_matrix)


def discover_cluster_bins(clusters_dir: str | Path) -> list[ClusterBin]:
    clusters_path = Path(clusters_dir).expanduser().resolve()
    if not clusters_path.is_dir():
        raise ValueError(f"Clusters directory does not exist: {clusters_path}")

    cluster_bins: list[ClusterBin] = []
    for structure_dir in sorted(
        clusters_path.iterdir(),
        key=lambda path: _natural_sort_key(path.name),
    ):
        if not structure_dir.is_dir():
            continue
        if structure_dir.name.startswith("representative_"):
            continue

        motif_dirs = sorted(
            [
                candidate
                for candidate in structure_dir.iterdir()
                if candidate.is_dir() and candidate.name.startswith("motif_")
            ],
            key=lambda path: _natural_sort_key(path.name),
        )

        if motif_dirs:
            for motif_dir in motif_dirs:
                files = _structure_files_in_dir(motif_dir)
                if not files:
                    continue
                cluster_bins.append(
                    ClusterBin(
                        structure=structure_dir.name,
                        motif=motif_dir.name,
                        source_dir=motif_dir,
                        files=tuple(files),
                        representative=files[0].name,
                    )
                )
            continue

        files = _structure_files_in_dir(structure_dir)
        if not files:
            continue
        cluster_bins.append(
            ClusterBin(
                structure=structure_dir.name,
                motif="no_motif",
                source_dir=structure_dir,
                files=tuple(files),
                representative=files[0].name,
            )
        )

    return cluster_bins


def discover_available_elements(clusters_dir: str | Path) -> list[str]:
    unique_elements: set[str] = set()
    for cluster_bin in discover_cluster_bins(clusters_dir):
        for file_path in cluster_bin.files:
            try:
                elements = scan_structure_elements(file_path)
            except Exception as exc:
                warnings.warn(
                    f"Skipping {file_path} while scanning available elements: "
                    f"{exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            unique_elements.update(
                _normalized_element_symbol(element)
                for element in elements
                if _normalized_element_symbol(element)
            )
    return sorted(unique_elements, key=_natural_sort_key)


def _structure_files_in_dir(directory: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdb", ".xyz"}
        ],
        key=lambda path: _natural_sort_key(path.name),
    )


class DebyeProfileBuilder:
    """Average Debye scattering profiles using the legacy
    SAXSClusterAverager calculation."""

    def __init__(
        self,
        *,
        q_values: np.ndarray,
        output_dir: str | Path,
        include_elements: list[str] | None = None,
        exclude_elements: list[str] | None = None,
    ) -> None:
        self.q_values = np.asarray(q_values, dtype=float)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.include_elements = {
            _normalized_element_symbol(value)
            for value in include_elements or []
        }
        self.exclude_elements = {
            _normalized_element_symbol(value)
            for value in exclude_elements or []
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_profiles(
        self,
        clusters_dir: str | Path | None = None,
        *,
        cluster_bins: list[ClusterBin] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        progress_offset: int = 0,
        progress_total: int | None = None,
    ) -> list[AveragedComponent]:
        if cluster_bins is None and clusters_dir is None:
            raise ValueError(
                "Either clusters_dir or cluster_bins must be provided."
            )
        active_cluster_bins = list(
            cluster_bins
            if cluster_bins is not None
            else discover_cluster_bins(clusters_dir)
        )
        total_files = sum(
            len(cluster_bin.files) for cluster_bin in active_cluster_bins
        )
        if progress_callback is not None:
            progress_callback(
                progress_offset,
                max(progress_total or total_files or 1, 1),
                "Preparing SAXS component calculation...",
            )

        components: list[AveragedComponent] = []
        processed_files = 0
        last_progress_report = -1
        for cluster_bin in active_cluster_bins:
            filtered_structures: list[tuple[np.ndarray, list[str]]] = []
            unique_elements: list[str] = []
            for file_path in cluster_bin.files:
                coords, elements = load_structure_file(file_path)
                filtered = self._filter_atoms(coords, elements)
                processed_files += 1
                if progress_callback is not None and _should_emit_progress(
                    processed_files,
                    total_files,
                    last_progress_report,
                ):
                    last_progress_report = processed_files
                    progress_callback(
                        progress_offset + processed_files,
                        max(progress_total or total_files or 1, 1),
                        (
                            "Building SAXS components: "
                            f"{cluster_bin.structure}/{cluster_bin.motif}"
                        ),
                    )
                if filtered is None:
                    continue
                filtered_coords, filtered_elements = filtered
                filtered_structures.append(
                    (filtered_coords, filtered_elements)
                )
                unique_elements.extend(filtered_elements)
            if not filtered_structures:
                continue

            f0_dictionary = build_f0_dictionary(unique_elements, self.q_values)
            traces = self._compute_cluster_traces(
                filtered_structures,
                f0_dictionary=f0_dictionary,
            )
            stacked = np.asarray(traces, dtype=float)
            mean_trace = stacked.mean(axis=0)
            std_trace = stacked.std(axis=0)
            se_trace = std_trace / np.sqrt(stacked.shape[0])
            output_path = self._write_component_file(
                cluster_bin.structure,
                cluster_bin.motif,
                mean_trace,
                std_trace,
                se_trace,
                len(traces),
            )
            components.append(
                AveragedComponent(
                    structure=cluster_bin.structure,
                    motif=cluster_bin.motif,
                    file_count=len(traces),
                    representative=cluster_bin.representative,
                    source_dir=cluster_bin.source_dir,
                    q_values=self.q_values.copy(),
                    mean_intensity=mean_trace,
                    std_intensity=std_trace,
                    se_intensity=se_trace,
                    output_path=output_path,
                )
            )
        if progress_callback is not None and progress_total is not None:
            progress_callback(
                progress_total,
                max(progress_total, 1),
                "SAXS component build complete.",
            )
        return components

    def _filter_atoms(
        self,
        coordinates: np.ndarray,
        elements: list[str],
    ) -> tuple[np.ndarray, list[str]] | None:
        return _filter_atoms_by_elements(
            coordinates,
            [_normalized_element_symbol(element) for element in elements],
            include_elements=self.include_elements or None,
            exclude_elements=self.exclude_elements or None,
        )

    def _compute_cluster_traces(
        self,
        filtered_structures: list[tuple[np.ndarray, list[str]]],
        *,
        f0_dictionary: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        if self._all_single_atom_structures_are_equivalent(
            filtered_structures
        ):
            filtered_coords, filtered_elements = filtered_structures[0]
            trace = compute_debye_intensity(
                filtered_coords,
                filtered_elements,
                self.q_values,
                f0_dictionary=f0_dictionary,
            )
            return [np.asarray(trace, dtype=float)] * len(filtered_structures)

        return [
            compute_debye_intensity(
                filtered_coords,
                filtered_elements,
                self.q_values,
                f0_dictionary=f0_dictionary,
            )
            for filtered_coords, filtered_elements in filtered_structures
        ]

    @staticmethod
    def _all_single_atom_structures_are_equivalent(
        filtered_structures: list[tuple[np.ndarray, list[str]]],
    ) -> bool:
        if not filtered_structures:
            return False
        atom_signatures: set[tuple[str, ...]] = set()
        for filtered_coords, filtered_elements in filtered_structures:
            if len(filtered_coords) != 1 or len(filtered_elements) != 1:
                return False
            atom_signatures.add(tuple(filtered_elements))
        return len(atom_signatures) == 1

    def _write_component_file(
        self,
        structure: str,
        motif: str,
        mean_trace: np.ndarray,
        std_trace: np.ndarray,
        se_trace: np.ndarray,
        n_structures: int,
    ) -> Path:
        safe_name = f"{structure}_{motif}".replace("/", "_")
        output_path = self.output_dir / f"{safe_name}.txt"
        header = (
            f"# Number of files: {n_structures}\n"
            "# Columns: q, S(q)_avg, S(q)_std, S(q)_se\n"
        )
        data = np.column_stack(
            [self.q_values, mean_trace, std_trace, se_trace]
        )
        np.savetxt(
            output_path,
            data,
            comments="",
            header=header,
            fmt=["%.8f", "%.8f", "%.8f", "%.8f"],
        )
        return output_path


def _should_emit_progress(
    processed: int,
    total: int,
    last_reported: int,
) -> bool:
    if processed <= 0:
        return True
    if total <= 0 or processed >= total:
        return True
    if total <= 200:
        return processed != last_reported
    interval = max(total // 200, 25)
    return processed - last_reported >= interval
