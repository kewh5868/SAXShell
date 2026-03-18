from __future__ import annotations

import csv
import os
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from string import ascii_uppercase
from typing import Dict, List, Optional, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Core configuration and small data containers
# ---------------------------------------------------------------------------


@dataclass
class XYZClusterConfig:
    """Configuration for XYZ-based solute cluster extraction.

    Parameters
    ----------
    node_elements : set of str
        Elements that act as "nodes" (e.g. {"Pb"}).
    linker_elements : set of str
        Elements that act as "linkers" (e.g. {"I"}).
    connectivity_cutoffs : dict[(str,str) -> float]
        Cutoffs for building the solute connectivity graph.
        Example: {("Pb","I"): 3.5}.
    coordination_cutoffs : dict[(str,str) -> float]
        Cutoffs for coordination counting.
        Example: {("Pb","O"): 3.0}.
    shell_radius : float
        Radius (Å) for spherical extraction around node centers.
    box : array-like of length 3, optional
        Simulation box lengths [Lx, Ly, Lz] for PBC. If None, no PBC.
    solute_elements : set of str, optional
        Elements considered part of the solute (defaults to node ∪ linker).
    solvent_donor_element : str, optional
        Element used to classify number of coordinated solvents, e.g. "O".
        For DMF-like systems, you can treat each coordinated O as one solvent.
    """

    node_elements: Set[str]
    linker_elements: Set[str]
    connectivity_cutoffs: Dict[Tuple[str, str], float]
    coordination_cutoffs: Dict[Tuple[str, str], float]
    shell_radius: float
    box: Optional[np.ndarray] = None
    solute_elements: Optional[Set[str]] = None
    solvent_donor_element: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize element symbols to Title case ("Pb", "I", "O")
        self.node_elements = {e.title() for e in self.node_elements}
        self.linker_elements = {e.title() for e in self.linker_elements}
        if self.solute_elements is None:
            self.solute_elements = set(self.node_elements) | set(
                self.linker_elements
            )
        else:
            self.solute_elements = {e.title() for e in self.solute_elements}

        # Normalize coordination + connectivity keys to Title case
        self.connectivity_cutoffs = {
            (a.title(), b.title()): float(r)
            for (a, b), r in self.connectivity_cutoffs.items()
        }
        self.coordination_cutoffs = {
            (a.title(), b.title()): float(r)
            for (a, b), r in self.coordination_cutoffs.items()
        }

        if self.box is not None:
            self.box = np.asarray(self.box, dtype=float)

        if self.solvent_donor_element is not None:
            self.solvent_donor_element = self.solvent_donor_element.title()

        if self.shell_radius <= 0:
            raise ValueError("shell_radius must be positive")

    @property
    def max_connectivity_cutoff(self) -> float:
        if not self.connectivity_cutoffs:
            return 0.0
        return max(self.connectivity_cutoffs.values())

    @property
    def max_coordination_cutoff(self) -> float:
        if not self.coordination_cutoffs:
            return 0.0
        return max(self.coordination_cutoffs.values())


@dataclass
class XYZFrame:
    """A single XYZ frame.

    Attributes
    ----------
    symbols : (N,) array of str
        Element symbols for each atom.
    coords : (N,3) ndarray
        Cartesian coordinates in Å (in simulation box).
    box : (3,) ndarray or None
        Box lengths [Lx, Ly, Lz]. If None, no PBC is applied.
    frame_id : str
        Identifier (e.g. file stem or "frame_0001").
    file_path : str
        Source XYZ file path for this frame.
    """

    symbols: np.ndarray
    coords: np.ndarray
    box: Optional[np.ndarray]
    frame_id: str
    file_path: str


@dataclass
class SoluteCluster:
    """A connected solute cluster (e.g. Pb2I4).

    Attributes
    ----------
    frame_id : str
    cluster_index : int
        Index local to the frame (1-based).
    atom_indices : list of int
        Indices of solute atoms in this cluster.
    node_indices : list of int
        Subset of atom_indices that are "nodes" (e.g. Pb).
    stoich_counts : dict[str,int]
        Element -> count for solute atoms in this cluster.
    stoich_label : str
        Formula-like label (e.g. "Pb2I4").
    virtual_positions : dict[int, np.ndarray]
        Virtual coordinates for atoms in this cluster's BFS space.
        - For solute atoms, set during connectivity BFS if PBC is used.
        - For shell atoms, set during shell extraction if PBC is used.
    """

    frame_id: str
    cluster_index: int
    atom_indices: List[int]
    node_indices: List[int]
    stoich_counts: Dict[str, int]
    stoich_label: str
    virtual_positions: Dict[int, np.ndarray] = field(default_factory=dict)


class XYZClusterExtractor:
    """Extract solute clusters + solvent shells frame-by-frame from XYZ
    files.

    - Identifies connected solute clusters using node/linker connectivity.
    - For each cluster:
        * extracts a spherical shell around node atoms, excluding solute atoms
          that are not part of that cluster.
        * counts coordinated solvent donors (e.g. O within Pb–O cutoff).
        * writes a per-cluster XYZ file.
    - Writes a cluster_metadata.csv summarizing all clusters.

    Cluster identity is always:
        (frame_id, cluster_index)

    Filenames are derived from:
        <frame_id>_cluster_<cluster_index>_<stoich_label>_solv<n>.xyz
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config: XYZClusterConfig,
        use_pbc: bool = False,
        debug_pbc: bool = False,
    ) -> None:
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.config = config
        self.use_pbc = bool(use_pbc)
        self.debug_pbc = bool(debug_pbc)

        os.makedirs(self.output_dir, exist_ok=True)

        self.metadata_rows: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------

    def _iter_xyz_files(self) -> List[str]:
        files = [
            os.path.join(self.input_dir, fn)
            for fn in os.listdir(self.input_dir)
            if fn.lower().endswith(".xyz")
        ]
        files.sort()
        return files

    def _load_xyz_single_frame(self, path: str) -> XYZFrame:
        """
        Minimal XYZ reader for single-frame files:
            line 1: N
            line 2: comment (ignored)
            next N lines: symbol x y z
        """
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        try:
            n_atoms = int(lines[0])
        except Exception as exc:
            raise ValueError(
                f"XYZ file {path} does not start with atom count"
            ) from exc

        if n_atoms <= 0 or len(lines) < 2 + n_atoms:
            raise ValueError(
                f"XYZ file {path} is too short for {n_atoms} atoms"
            )

        data_lines = lines[2 : 2 + n_atoms]
        symbols: List[str] = []
        coords: List[List[float]] = []

        for ln in data_lines:
            parts = ln.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ line in {path}: {ln}")
            sym = parts[0].title()
            x, y, z = map(float, parts[1:4])
            symbols.append(sym)
            coords.append([x, y, z])

        symbols_arr = np.array(symbols, dtype=object)
        coords_arr = np.array(coords, dtype=float)

        frame_id = os.path.splitext(os.path.basename(path))[0]
        box = self.config.box.copy() if self.config.box is not None else None

        return XYZFrame(
            symbols=symbols_arr,
            coords=coords_arr,
            box=box,
            frame_id=frame_id,
            file_path=path,
        )

    # ------------------------------------------------------------------
    # Distance and PBC helpers
    # ------------------------------------------------------------------

    def _min_image_delta(self, dr: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Apply minimum-image convention to a displacement vector dr
        under PBC."""
        return dr - np.round(dr / box) * box

    def _distance(
        self, r1: np.ndarray, r2: np.ndarray, box: Optional[np.ndarray]
    ) -> float:
        """Distance between two points, with optional PBC."""
        dr = r2 - r1
        if self.use_pbc and box is not None:
            dr = self._min_image_delta(dr, box)
        return float(np.linalg.norm(dr))

    # ------------------------------------------------------------------
    # Connectivity and cluster finding
    # ------------------------------------------------------------------

    def _connectivity_cutoff(
        self, elem_a: str, elem_b: str
    ) -> Optional[float]:
        pair = (elem_a, elem_b)
        rev = (elem_b, elem_a)
        if pair in self.config.connectivity_cutoffs:
            return self.config.connectivity_cutoffs[pair]
        if rev in self.config.connectivity_cutoffs:
            return self.config.connectivity_cutoffs[rev]
        return None

    def _build_solute_connectivity(
        self, frame: XYZFrame
    ) -> Dict[int, List[int]]:
        """Build adjacency list for solute atoms based on
        connectivity_cutoffs."""
        symbols = frame.symbols
        coords = frame.coords
        box = frame.box

        solute_mask = np.array(
            [sym in self.config.solute_elements for sym in symbols]
        )
        solute_indices = np.where(solute_mask)[0]
        solute_coords = coords[solute_mask]

        if solute_indices.size == 0:
            return {}

        # KDTree in either full box or just Euclidean space
        tree = KDTree(solute_coords)
        max_cut = self.config.max_connectivity_cutoff
        adjacency: Dict[int, List[int]] = {int(i): [] for i in solute_indices}

        for idx_local, i in enumerate(solute_indices):
            neighbors_local = tree.query_ball_point(
                solute_coords[idx_local], r=max_cut
            )
            for j_local in neighbors_local:
                if j_local == idx_local:
                    continue
                j = solute_indices[j_local]
                elem_i = symbols[i]
                elem_j = symbols[j]
                cutoff_ij = self._connectivity_cutoff(elem_i, elem_j)
                if cutoff_ij is None:
                    continue
                d = self._distance(coords[i], coords[j], box)
                if d <= cutoff_ij:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        return adjacency

    def _build_stoich_label(self, stoich_counts: Dict[str, int]) -> str:
        """Turn a dict like {"Pb": 2, "I": 4} into 'Pb2I4', with Pb–I
        preference."""
        counts = {k: v for k, v in stoich_counts.items() if v > 0}
        if not counts:
            return "None"

        pieces: List[str] = []

        # Pb–I preference if present
        if "Pb" in counts:
            n = counts.pop("Pb")
            pieces.append("Pb" if n == 1 else f"Pb{n}")
        if "I" in counts:
            n = counts.pop("I")
            pieces.append("I" if n == 1 else f"I{n}")

        for elem in sorted(counts.keys()):
            n = counts[elem]
            pieces.append(elem if n == 1 else f"{elem}{n}")

        return "".join(pieces)

    def _find_solute_clusters(self, frame: XYZFrame) -> List[SoluteCluster]:
        """Find connected solute clusters in the frame."""
        adjacency = self._build_solute_connectivity(frame)
        if not adjacency:
            return []

        visited: Set[int] = set()
        clusters: List[SoluteCluster] = []
        coords = frame.coords
        box = frame.box

        virtual_positions: Dict[int, np.ndarray] = {}

        def bfs(start_atom: int, cluster_index: int) -> SoluteCluster:
            queue = [start_atom]
            visited_local: Set[int] = set([start_atom])
            visited.add(start_atom)

            if self.use_pbc and box is not None:
                virtual_positions[start_atom] = coords[start_atom].copy()

            while queue:
                i = queue.pop(0)
                ri = coords[i]
                for j in adjacency[i]:
                    if j in visited:
                        continue
                    rj = coords[j]
                    if self.use_pbc and box is not None:
                        dr = rj - ri
                        dr = self._min_image_delta(dr, box)
                        vj = virtual_positions.get(i, ri) + dr
                        virtual_positions[j] = vj
                    visited.add(j)
                    visited_local.add(j)
                    queue.append(j)

            atom_indices = sorted(list(visited_local))
            node_indices = [
                idx
                for idx in atom_indices
                if frame.symbols[idx] in self.config.node_elements
            ]
            stoich_counts: Dict[str, int] = {}
            for idx in atom_indices:
                elem = frame.symbols[idx]
                stoich_counts[elem] = stoich_counts.get(elem, 0) + 1

            stoich_label = self._build_stoich_label(stoich_counts)

            # Extract virtual positions only for this cluster
            cluster_vpos = {
                idx: virtual_positions[idx]
                for idx in atom_indices
                if idx in virtual_positions
            }

            return SoluteCluster(
                frame_id=frame.frame_id,
                cluster_index=cluster_index,
                atom_indices=atom_indices,
                node_indices=node_indices,
                stoich_counts=stoich_counts,
                stoich_label=stoich_label,
                virtual_positions=cluster_vpos,
            )

        cluster_counter = 0
        for atom_idx in sorted(adjacency.keys()):
            if atom_idx in visited:
                continue
            cluster_counter += 1
            cluster = bfs(atom_idx, cluster_counter)
            clusters.append(cluster)

        return clusters

    # ------------------------------------------------------------------
    # Coordination counting
    # ------------------------------------------------------------------

    def _coordination_cutoff(
        self, elem_center: str, elem_neighbor: str
    ) -> Optional[float]:
        pair = (elem_center, elem_neighbor)
        rev = (elem_neighbor, elem_center)
        if pair in self.config.coordination_cutoffs:
            return self.config.coordination_cutoffs[pair]
        if rev in self.config.coordination_cutoffs:
            return self.config.coordination_cutoffs[rev]
        return None

    def _count_solv_donors_for_cluster(
        self, frame: XYZFrame, cluster: SoluteCluster
    ) -> Tuple[int, Dict[int, int]]:
        """Count coordinated solvent donors (e.g. O) per node center in
        this cluster.

        Returns
        -------
        total_donors : int
            Sum of all donors across node centers (you can interpret as # of solvent).
        per_center : dict[int, int]
            center_index -> count of donor atoms within cutoff.
        """
        symbols = frame.symbols
        coords = frame.coords
        box = frame.box

        donor_elem = self.config.solvent_donor_element
        if donor_elem is None:
            return 0, {}

        donor_indices = [
            i for i, sym in enumerate(symbols) if sym == donor_elem
        ]
        if not donor_indices:
            return 0, {}

        donor_indices_arr = np.array(donor_indices, dtype=int)
        donor_coords = coords[donor_indices_arr]
        tree = KDTree(donor_coords)

        max_cut = self.config.max_coordination_cutoff
        per_center: Dict[int, int] = {}

        for center_idx in cluster.node_indices:
            center_elem = symbols[center_idx]
            cutoff = self._coordination_cutoff(center_elem, donor_elem)
            if cutoff is None:
                continue

            rc = coords[center_idx]

            if self.use_pbc and box is not None:
                # brute-force KDTree + min-image refinement
                candidate_ids = tree.query_ball_point(rc, r=cutoff + 2.0)
                count = 0
                for cid in candidate_ids:
                    j = donor_indices_arr[cid]
                    rj = coords[j]
                    d = self._distance(rc, rj, box)
                    if d <= cutoff:
                        count += 1
                per_center[center_idx] = count
            else:
                # Euclidean sphere around center
                candidate_ids = tree.query_ball_point(rc, r=cutoff)
                per_center[center_idx] = len(candidate_ids)

        total_donors = sum(per_center.values())
        return total_donors, per_center

    # ------------------------------------------------------------------
    # Shell extraction
    # ------------------------------------------------------------------

    def _extract_shell_indices(
        self, frame: XYZFrame, cluster: SoluteCluster
    ) -> Set[int]:
        """
        Extract indices to include in the cluster XYZ:
          - all solute atoms belonging to this cluster
          - all atoms (solute or solvent) within shell_radius of any node center
            EXCEPT solute atoms that belong to other clusters.

        Solvent atoms are allowed to be reused in different clusters.
        """
        symbols = frame.symbols
        coords = frame.coords
        box = frame.box
        radius = self.config.shell_radius

        extracted: Set[int] = set(cluster.atom_indices)

        if not cluster.node_indices:
            return extracted

        # Precompute which indices are solute (of any cluster)
        solute_mask = np.array(
            [sym in self.config.solute_elements for sym in symbols]
        )
        all_solute_indices = set(np.where(solute_mask)[0])

        for center_idx in cluster.node_indices:
            rc = coords[center_idx]
            for j in range(coords.shape[0]):
                # Exclude solute atoms not in this cluster
                if j in all_solute_indices and j not in cluster.atom_indices:
                    continue
                d = self._distance(rc, coords[j], box)
                if d <= radius:
                    extracted.add(j)

        return extracted

    # ------------------------------------------------------------------
    # File output + metadata
    # ------------------------------------------------------------------

    def _coord_per_center_string(self, per_center: Dict[int, int]) -> str:
        """Build an Excel-safe string describing coordination per
        center.

        Example: center150[O=1];center151[O=2]
        """
        if not per_center:
            return ""
        parts: List[str] = []
        for center_idx in sorted(per_center.keys()):
            count = per_center[center_idx]
            parts.append(f"center{center_idx}[O={count}]")
        return ";".join(parts)

    def _write_cluster_xyz(
        self,
        frame: XYZFrame,
        cluster: SoluteCluster,
        extracted_indices: Set[int],
        n_solv_donors: int,
    ) -> str:
        """Write an XYZ file for this cluster.

        Folder structure:
            <output_dir>/
                <stoich_label>/             e.g. "Pb2I4"
                    solv_<n_solv_donors>/   e.g. "solv_2"
                        <frame>_cluster_<idx>_<stoich>_solv<n>.xyz

        Coordinates are written in the original box frame if use_pbc=False,
        or in the cluster's virtual frame if use_pbc=True and virtual positions exist.
        """
        indices_sorted = sorted(extracted_indices)
        symbols = frame.symbols
        coords = frame.coords

        # Decide which coordinates to write
        if self.use_pbc and cluster.virtual_positions:
            xyz_coords = []
            for idx in indices_sorted:
                if idx in cluster.virtual_positions:
                    xyz_coords.append(cluster.virtual_positions[idx])
                else:
                    xyz_coords.append(coords[idx])
            xyz_coords = np.asarray(xyz_coords, dtype=float)
        else:
            xyz_coords = coords[indices_sorted]

        # Build folder structure: <output_dir>/<stoich_label>/solv_<n>/
        stoich_dir = os.path.join(self.output_dir, cluster.stoich_label)
        solv_dir = os.path.join(stoich_dir, f"solv_{n_solv_donors}")
        os.makedirs(solv_dir, exist_ok=True)

        # File stem
        stem = (
            f"{frame.frame_id}_cluster_{cluster.cluster_index:03d}_"
            f"{cluster.stoich_label}_solv{n_solv_donors}"
        )
        out_path = os.path.join(solv_dir, stem + ".xyz")

        # Write XYZ
        with open(out_path, "w") as f:
            f.write(f"{len(indices_sorted)}\n")
            f.write(
                f"frame={frame.frame_id} cluster={cluster.cluster_index} "
                f"stoich={cluster.stoich_label} n_solv_donors={n_solv_donors}\n"
            )
            for idx, coord in zip(indices_sorted, xyz_coords):
                x, y, z = coord
                f.write(f"{symbols[idx]} {x:.6f} {y:.6f} {z:.6f}\n")

        return out_path

    def _build_metadata_row(
        self,
        frame: XYZFrame,
        cluster: SoluteCluster,
        extracted_indices: Set[int],
        n_solv_donors: int,
        per_center: Dict[int, int],
        xyz_path: str,
    ) -> Dict[str, object]:
        """Build a dict for one row of cluster_metadata.

        Includes:
          - relative stoichiometry folder (stoich_dir_rel)
          - relative solvent folder (solv_dir_rel)
        so that external tools can reconstruct the folder structure.
        """
        row: Dict[str, object] = {}
        row["frame_id"] = frame.frame_id
        row["cluster_index"] = cluster.cluster_index
        row["source_file"] = frame.file_path
        row["stoich_label"] = cluster.stoich_label
        row["n_solute_atoms"] = len(cluster.atom_indices)
        row["n_node_centers"] = len(cluster.node_indices)
        row["n_atoms_extracted"] = len(extracted_indices)
        row["n_solv_donors"] = n_solv_donors
        row["xyz_path"] = xyz_path

        # Folder structure relative to output_dir
        stoich_dir_rel = cluster.stoich_label
        solv_dir_rel = os.path.join(
            cluster.stoich_label, f"solv_{n_solv_donors}"
        )
        row["stoich_dir_rel"] = stoich_dir_rel
        row["solv_dir_rel"] = solv_dir_rel

        # Element-resolved solute counts (for histogram analyzer)
        for elem, count in cluster.stoich_counts.items():
            row[f"solute_{elem}"] = int(count)

        # Coordination per center, Excel-safe
        row["coordination_per_center"] = self._coord_per_center_string(
            per_center
        )

        return row

    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Process all XYZ files in input_dir and write:
          - One XYZ file per extracted cluster.
          - cluster_metadata.csv in output_dir.

        No three-letter cluster IDs are used; cluster identity is
        (frame_id, cluster_index) only.
        """
        xyz_files = self._iter_xyz_files()
        if not xyz_files:
            print(
                f"[XYZClusterExtractor] No .xyz files found in {self.input_dir}"
            )
            return

        for path in xyz_files:
            frame = self._load_xyz_single_frame(path)
            clusters = self._find_solute_clusters(frame)
            if not clusters:
                continue

            for cluster in clusters:
                # Count solvent donors coordinated to this cluster
                n_solv_donors, per_center = (
                    self._count_solv_donors_for_cluster(frame, cluster)
                )

                # Extract shell (solute + surrounding solvent)
                extracted_indices = self._extract_shell_indices(frame, cluster)

                # Write cluster XYZ
                xyz_path = self._write_cluster_xyz(
                    frame=frame,
                    cluster=cluster,
                    extracted_indices=extracted_indices,
                    n_solv_donors=n_solv_donors,
                )

                # Metadata row
                row = self._build_metadata_row(
                    frame=frame,
                    cluster=cluster,
                    extracted_indices=extracted_indices,
                    n_solv_donors=n_solv_donors,
                    per_center=per_center,
                    xyz_path=xyz_path,
                )
                self.metadata_rows.append(row)

        # Write metadata CSV
        if self.metadata_rows:
            df = pd.DataFrame(self.metadata_rows)
            df.sort_values(
                by=["frame_id", "cluster_index"],
                inplace=True,
                ignore_index=True,
            )
            meta_path = os.path.join(self.output_dir, "cluster_metadata.csv")
            df.to_csv(meta_path, index=False)
            print(f"[XYZClusterExtractor] Wrote metadata: {meta_path}")
        else:
            print(
                "[XYZClusterExtractor] No clusters extracted; no metadata written."
            )


class ClusterHistogramAnalyzer:
    """Analyze cluster_metadata.csv from XYZClusterExtractor and build
    histograms of solute stoichiometries vs solvent coordination.

    Requires metadata CSV with at least:
      - stoich_label     (e.g. 'Pb2I4')
      - n_solv_donors

    Stoichiometry parsing
    ---------------------
    stoich_label is parsed with a regex:
      'Pb2I4' -> {'Pb': 2, 'I': 4}

    From this we derive per cluster:
      - node_count         (# of node_element atoms, e.g. Pb)
      - anion_count        (# of anion_element atoms, e.g. I)
      - solute_atom_count  (sum of all elements in the stoich label)

    Weighting options
    -----------------
    weighting = "centers":
        Each cluster weighted by node_count (Pb centers).
    weighting = "clusters":
        Each cluster has weight 1.
    weighting = "solute_atoms":
        Each cluster weighted by solute_atom_count (Pb + I + ...).

    y_mode options
    --------------
    y_mode = "count":
        Raw weighted counts per (stoich_label, n_solv_donors) bin.
    y_mode = "fraction":
        Counts normalized by the total weight over all clusters (sum of all bins = 1).
    y_mode = "percent":
        Same as "fraction" but multiplied by 100.
    """

    def __init__(
        self,
        output_dir: str,
        node_element: str = "Pb",
        anion_element: str = "I",
        metadata_file: str = "cluster_metadata.csv",
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.node_element = node_element
        self.anion_element = anion_element
        self.metadata_path = os.path.join(self.output_dir, metadata_file)

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata CSV not found at: {self.metadata_path}"
            )

        self.df = pd.read_csv(self.metadata_path)

        if "stoich_label" not in self.df.columns:
            raise ValueError("metadata CSV is missing 'stoich_label' column")
        if "n_solv_donors" not in self.df.columns:
            raise ValueError("metadata CSV is missing 'n_solv_donors' column")

        # Parse stoich_label -> element counts
        parsed = (
            self.df["stoich_label"].astype(str).apply(self._parse_stoich_label)
        )

        # Number of Pb centers, number of I, and total solute atoms
        self.df["node_count"] = parsed.apply(
            lambda d: int(d.get(self.node_element, 0))
        )
        self.df["anion_count"] = parsed.apply(
            lambda d: int(d.get(self.anion_element, 0))
        )
        self.df["solute_atom_count"] = parsed.apply(
            lambda d: int(sum(d.values()))
        )

    # ------------------------------------------------------------------
    # Stoichiometry parser
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_stoich_label(label: str) -> dict:
        """Parse a stoichiometry string like 'Pb2I4' into a dict: {'Pb':
        2, 'I': 4}. Missing numbers are treated as 1, e.g. 'PbI2' ->
        {'Pb': 1, 'I': 2}.

        Note: any leading digits (e.g. '4I2Pb') are ignored by design;
        we assume labels are chemical-style, not ratio-coded.
        """
        tokens = re.findall(r"([A-Z][a-z]*)(\d*)", label)
        counts = {}
        for elem, num in tokens:
            n = int(num) if num else 1
            counts[elem] = counts.get(elem, 0) + n
        return counts

    # ------------------------------------------------------------------
    # Axis-label formatter (Pb–I special case)
    # ------------------------------------------------------------------
    def _format_stoich_for_axis(self, label: str) -> str:
        """Format a stoich_label for display on the x-axis with
        subscripts.

        Pb–I special case:
          - Order is always Pb then I (if present), then any other elements.
          - Uses mathtext subscripts, e.g. 'Pb2I4' -> 'Pb$_2$I$_4$'.
        Generic case:
          - Elements sorted alphabetically with subscripts.
        """
        counts = self._parse_stoich_label(label)

        def fmt_elem(elem: str, n: int) -> str:
            if n <= 0:
                return ""
            if n == 1:
                return elem
            return f"{elem}$_{{{n}}}$"

        # Pb–I special case
        if self.node_element == "Pb" and self.anion_element == "I":
            n_pb = counts.get("Pb", 0)
            n_i = counts.get("I", 0)

            pieces: List[str] = []
            pieces.append(fmt_elem("Pb", n_pb))
            pieces.append(fmt_elem("I", n_i))

            # Any other elements appended after Pb / I in alphabetical order
            for elem in sorted(counts.keys()):
                if elem in {"Pb", "I"}:
                    continue
                pieces.append(fmt_elem(elem, counts[elem]))

            text = "".join(pieces)
            return text if text else label

        # Generic: alphabetical
        pieces = []
        for elem in sorted(counts.keys()):
            pieces.append(fmt_elem(elem, counts[elem]))
        text = "".join(pieces)
        return text if text else label

    # ------------------------------------------------------------------
    # Core distribution computation
    # ------------------------------------------------------------------
    def compute_distribution(
        self,
        weighting: Literal["centers", "clusters", "solute_atoms"] = "centers",
        y_mode: Literal["count", "fraction", "percent"] = "fraction",
    ) -> pd.DataFrame:
        """Build a tidy DataFrame representing the histogram
        distribution.

        Returns a DataFrame with columns:
          - stoich_label
          - n_solv_donors
          - weight         (raw weighted count)
          - value          (depending on y_mode)
        """
        df = self.df.copy()

        if weighting not in {"centers", "clusters", "solute_atoms"}:
            raise ValueError(
                "weighting must be 'centers', 'clusters', or 'solute_atoms'"
            )
        if y_mode not in {"count", "fraction", "percent"}:
            raise ValueError(
                "y_mode must be 'count', 'fraction', or 'percent'"
            )

        # Base weight
        if weighting == "centers":
            df["base_weight"] = df["node_count"].astype(float)
        elif weighting == "clusters":
            df["base_weight"] = 1.0
        else:  # "solute_atoms"
            df["base_weight"] = df["solute_atom_count"].astype(float)

        grouped = (
            df.groupby(["stoich_label", "n_solv_donors"], dropna=False)[
                "base_weight"
            ]
            .sum()
            .reset_index()
            .rename(columns={"base_weight": "weight"})
        )

        total_weight = grouped["weight"].sum()

        if y_mode == "count":
            grouped["value"] = grouped["weight"]
        elif y_mode == "fraction":
            grouped["value"] = (
                grouped["weight"] / total_weight if total_weight > 0 else 0.0
            )
        else:  # "percent"
            grouped["value"] = (
                100.0 * grouped["weight"] / total_weight
                if total_weight > 0
                else 0.0
            )

        return grouped

    # ------------------------------------------------------------------
    # Pb:I ordering helper
    # ------------------------------------------------------------------
    def _pb_i_sort_order(self) -> pd.DataFrame:
        """
        Build a small table with stoich_label, node_count, anion_count and
        a sort key enforcing:

          1) I-only (no Pb), sorted by increasing I count
          2) Pb-only (no I), sorted by increasing Pb count
          3) Mixed Pb–I, sorted by increasing Pb then I
        """
        cols = ["stoich_label", "node_count", "anion_count"]
        base = self.df[cols].drop_duplicates(subset=["stoich_label"]).copy()
        base = base.set_index("stoich_label")

        def sort_key(row) -> Tuple[int, int, int]:
            n_pb = int(row["node_count"])
            n_i = int(row["anion_count"])
            if n_pb == 0 and n_i > 0:
                group = 0  # I only
            elif n_pb > 0 and n_i == 0:
                group = 1  # Pb only
            else:
                group = 2  # mixed
            return (group, n_pb, n_i)

        base["sort_key"] = [sort_key(row) for _, row in base.iterrows()]
        return base

    # ------------------------------------------------------------------
    # Plotting + export
    # ------------------------------------------------------------------
    # def plot_distribution(
    #     self,
    #     weighting: Literal["centers", "clusters", "solute_atoms"] = "centers",
    #     y_mode: Literal["count", "fraction", "percent"] = "fraction",
    #     ax: Optional[plt.Axes] = None,
    #     figsize: Tuple[float, float] = (10, 3),
    #     color_scheme: str = "green",
    #     solvent_label: str = "solvent",       # e.g. "DMF", "DMSO"
    #     save_table: bool = False,
    #     table_filename: Optional[str] = None,
    #     show: bool = True,
    # ) -> Tuple[pd.DataFrame, plt.Axes]:
    #     """
    #     Plot a stacked bar chart:

    #       x-axis: stoich_label (ordered by Pb:I pattern)
    #       stacked bars: n_solv_donors (0, 1, 2, ...)

    #     Bars are annotated:
    #       - y_mode in {'fraction','percent'} → label as % above bar
    #         (bars < 2% get a red dot instead of text).
    #       - y_mode == 'count' → label with counts.

    #     color_scheme:
    #       - "green" (default): pale + green stack
    #       - any other string is treated as a matplotlib colormap name
    #         (e.g. "Blues", "viridis", ...).

    #     solvent_label:
    #       - Used in the legend text, e.g. "0 DMF", "2 DMF" instead of "0 solvent".

    #     save_table:
    #       - If True, writes a CSV into output_dir with one row per stoichiometry
    #         in the same x-axis order, columns:
    #           stoich_label, axis_label, order_index, and one column per
    #           n_solv_donors.

    #     table_filename:
    #       - Optional file name for the CSV. If None, a name is generated from
    #         weighting and y_mode.

    #     Returns
    #     -------
    #     dist_df : DataFrame
    #         The computed distribution DataFrame.
    #     ax : matplotlib.axes.Axes
    #         The axis with the plot.
    #     """
    #     # Use Arial everywhere for this plot
    #     mpl.rcParams["font.family"] = "Arial"

    #     dist_df = self.compute_distribution(weighting=weighting, y_mode=y_mode)

    #     # Pivot for stacked bar plot
    #     pivot = dist_df.pivot_table(
    #         index="stoich_label",
    #         columns="n_solv_donors",
    #         values="value",
    #         fill_value=0.0,
    #     )

    #     # Pb:I sort order
    #     order_df = self._pb_i_sort_order()
    #     common_labels = [lab for lab in order_df.index if lab in pivot.index]
    #     sorted_labels = sorted(
    #         common_labels,
    #         key=lambda lab: order_df.loc[lab, "sort_key"]
    #     )
    #     pivot = pivot.loc[sorted_labels]

    #     stoich_labels: List[str] = list(pivot.index)
    #     solv_levels: List[int] = list(pivot.columns)

    #     # ------------------------------------------------------------------
    #     # Optional export of table for external plotting
    #     # ------------------------------------------------------------------
    #     if save_table:
    #         table_df = pivot.copy().reset_index()  # stoich_label becomes column
    #         table_df.insert(1, "order_index", np.arange(len(table_df)))
    #         table_df.insert(
    #             2,
    #             "axis_label",
    #             [self._format_stoich_for_axis(lab) for lab in table_df["stoich_label"]],
    #         )

    #         # Rename solvent-count columns to something explicit: <solvent_label>_<n>
    #         rename_map = {}
    #         for col in table_df.columns:
    #             if isinstance(col, (int, np.integer)):
    #                 rename_map[col] = f"{solvent_label}_{int(col)}"
    #         table_df = table_df.rename(columns=rename_map)

    #         if table_filename is None:
    #             table_filename = f"histogram_table_{weighting}_{y_mode}.csv"
    #         out_path = os.path.join(self.output_dir, table_filename)
    #         table_df.to_csv(out_path, index=False)
    #         print(f"Histogram table written to: {out_path}")

    #     # ------------------------------------------------------------------
    #     # Plotting
    #     # ------------------------------------------------------------------
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)

    #     x = np.arange(len(stoich_labels))
    #     width = 0.8

    #     # Color scheme
    #     nonzero_levels = [lvl for lvl in sorted(solv_levels) if lvl != 0]
    #     n_nonzero = max(len(nonzero_levels), 1)

    #     if color_scheme == "green":
    #         zero_color = "#f6e8c3"
    #         green_palette = ["#00441b", "#1b7837", "#5aae61", "#a6d96a", "#d9f0d3"]
    #         stack_colors = [
    #             green_palette[min(i, len(green_palette) - 1)]
    #             for i in range(n_nonzero)
    #         ]
    #     else:
    #         cmap = plt.get_cmap(color_scheme)
    #         zero_color = cmap(0.1)
    #         stack_colors = [
    #             cmap(0.3 + 0.6 * i / max(n_nonzero - 1, 1))
    #             for i in range(n_nonzero)
    #         ]

    #     bottom = np.zeros(len(stoich_labels))

    #     # n_solv_donors == 0 first (if present)
    #     if 0 in pivot.columns:
    #         values = pivot[0].values
    #         ax.bar(
    #             x,
    #             values,
    #             width,
    #             bottom=bottom,
    #             label=f"0 {solvent_label}",
    #             color=zero_color,
    #             edgecolor="black",
    #             linewidth=0.5,
    #         )
    #         bottom = bottom + values

    #     # Then n_solv_donors > 0
    #     for i, solv in enumerate(nonzero_levels):
    #         values = pivot[solv].values
    #         color = stack_colors[i]
    #         ax.bar(
    #             x,
    #             values,
    #             width,
    #             bottom=bottom,
    #             label=f"{solv} {solvent_label}",
    #             color=color,
    #             edgecolor="black",
    #             linewidth=0.5,
    #         )
    #         bottom = bottom + values

    #     # Annotation: totals + red dots
    #     totals = pivot.sum(axis=1).values
    #     ymax = totals.max() if totals.size > 0 else 1.0
    #     if ymax <= 0:
    #         ymax = 1.0
    #     ax.set_ylim(0, ymax * 1.15)

    #     for i, label in enumerate(stoich_labels):
    #         total_val = totals[i]
    #         if total_val <= 0:
    #             continue

    #         if y_mode == "count":
    #             txt = f"{total_val:.0f}"
    #             ax.text(
    #                 x[i],
    #                 total_val * 1.01,
    #                 txt,
    #                 ha="center",
    #                 va="bottom",
    #                 fontsize=8,
    #             )
    #         else:
    #             if y_mode == "percent":
    #                 pct = total_val
    #             else:  # fraction
    #                 pct = total_val * 100.0

    #             if pct >= 2.0:
    #                 ax.text(
    #                     x[i],
    #                     total_val * 1.01,
    #                     f"{pct:.1f}%",
    #                     ha="center",
    #                     va="bottom",
    #                     fontsize=8,
    #                 )
    #             else:
    #                 ax.scatter(
    #                     x[i],
    #                     total_val,
    #                     color="red",
    #                     s=15,
    #                     zorder=4,
    #                 )

    #     # Axis labels, title, legend
    #     axis_labels = [self._format_stoich_for_axis(lab) for lab in stoich_labels]
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(axis_labels, rotation=45, ha="right")

    #     if weighting == "centers":
    #         base_label = f"{self.node_element} centers"
    #     elif weighting == "clusters":
    #         base_label = "clusters"
    #     else:
    #         base_label = "solute atoms"

    #     if y_mode == "count":
    #         ax.set_ylabel(f"Count of {base_label}")
    #     elif y_mode == "fraction":
    #         ax.set_ylabel(f"Fraction of {base_label}")
    #     else:
    #         ax.set_ylabel(f"Percentage of {base_label} [%]")

    #     if weighting == "centers":
    #         weight_desc = f"{self.node_element}-center-weighted"
    #     elif weighting == "clusters":
    #         weight_desc = "cluster-weighted"
    #     else:
    #         weight_desc = "solute-atom-weighted"

    #     ax.set_xlabel(f"{self.node_element}:{self.anion_element} structure")
    #     ax.set_title(f"Structure-Fraction Prior Histogram\n({weight_desc}, y={y_mode})")

    #     ax.legend(
    #         title=f"{solvent_label} coordination",
    #         fontsize="small",
    #         title_fontsize="small",
    #     )
    #     ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    #     plt.tight_layout()
    #     if show:
    #         plt.show()

    #     return dist_df, ax

    def plot_distribution(
        self,
        weighting: Literal["centers", "clusters", "solute_atoms"] = "centers",
        y_mode: Literal["count", "fraction", "percent"] = "fraction",
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (10, 3),
        color_scheme: str = "green",
        solvent_label: str = "solvent",
        save_table: bool = False,
        table_filename: Optional[str] = None,
        save_txt: bool = False,
        txt_filename: Optional[str] = None,
        txt_split_files: bool = False,
        txt_format: Literal["blocks", "flat"] = "blocks",
        show: bool = True,
    ) -> Tuple[pd.DataFrame, plt.Axes]:
        """Plot a stacked bar chart and optionally export tables.

        New export options
        ------------------
        save_txt:
            If True, writes a .txt export into output_dir.
        txt_filename:
            Optional name for the .txt file (single-file case).
        txt_split_files:
            If True, writes one .txt per stoichiometry.
        txt_format:
            "blocks" (human readable per-stoich sections) or "flat" (tab-delimited table)
        """
        mpl.rcParams["font.family"] = "Arial"

        dist_df = self.compute_distribution(weighting=weighting, y_mode=y_mode)

        pivot = dist_df.pivot_table(
            index="stoich_label",
            columns="n_solv_donors",
            values="value",
            fill_value=0.0,
        )

        order_df = self._pb_i_sort_order()
        common_labels = [lab for lab in order_df.index if lab in pivot.index]
        sorted_labels = sorted(
            common_labels, key=lambda lab: order_df.loc[lab, "sort_key"]
        )
        pivot = pivot.loc[sorted_labels]

        stoich_labels: List[str] = list(pivot.index)
        solv_levels: List[int] = list(pivot.columns)

        # Optional CSV export (existing)
        if save_table:
            table_df = pivot.copy().reset_index()
            table_df.insert(1, "order_index", np.arange(len(table_df)))
            table_df.insert(
                2,
                "axis_label",
                [
                    self._format_stoich_for_axis(lab)
                    for lab in table_df["stoich_label"]
                ],
            )

            rename_map = {}
            for col in table_df.columns:
                if isinstance(col, (int, np.integer)):
                    rename_map[col] = f"{solvent_label}_{int(col)}"
            table_df = table_df.rename(columns=rename_map)

            if table_filename is None:
                table_filename = f"histogram_table_{weighting}_{y_mode}.csv"
            out_path = os.path.join(self.output_dir, table_filename)
            table_df.to_csv(out_path, index=False)
            print(f"Histogram table written to: {out_path}")

        # Optional TXT export (new)
        if save_txt:
            out_paths = self.export_distribution_txt(
                weighting=weighting,
                y_mode=y_mode,
                solvent_label=solvent_label,
                filename=txt_filename,
                split_files=txt_split_files,
                fmt=txt_format,
            )
            for p in out_paths:
                print(f"Histogram TXT written to: {p}")

        # Plotting (existing)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(stoich_labels))
        width = 0.8

        nonzero_levels = [lvl for lvl in sorted(solv_levels) if lvl != 0]
        n_nonzero = max(len(nonzero_levels), 1)

        if color_scheme == "green":
            zero_color = "#f6e8c3"
            green_palette = [
                "#00441b",
                "#1b7837",
                "#5aae61",
                "#a6d96a",
                "#d9f0d3",
            ]
            stack_colors = [
                green_palette[min(i, len(green_palette) - 1)]
                for i in range(n_nonzero)
            ]
        else:
            cmap = plt.get_cmap(color_scheme)
            zero_color = cmap(0.1)
            stack_colors = [
                cmap(0.3 + 0.6 * i / max(n_nonzero - 1, 1))
                for i in range(n_nonzero)
            ]

        bottom = np.zeros(len(stoich_labels))

        if 0 in pivot.columns:
            values = pivot[0].values
            ax.bar(
                x,
                values,
                width,
                bottom=bottom,
                label=f"0 {solvent_label}",
                color=zero_color,
                edgecolor="black",
                linewidth=0.5,
            )
            bottom = bottom + values

        for i, solv in enumerate(nonzero_levels):
            values = pivot[solv].values
            color = stack_colors[i]
            ax.bar(
                x,
                values,
                width,
                bottom=bottom,
                label=f"{solv} {solvent_label}",
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            bottom = bottom + values

        totals = pivot.sum(axis=1).values
        ymax = totals.max() if totals.size > 0 else 1.0
        if ymax <= 0:
            ymax = 1.0
        ax.set_ylim(0, ymax * 1.15)

        for i, label in enumerate(stoich_labels):
            total_val = totals[i]
            if total_val <= 0:
                continue

            if y_mode == "count":
                txt = f"{total_val:.0f}"
                ax.text(
                    x[i],
                    total_val * 1.01,
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            else:
                if y_mode == "percent":
                    pct = total_val
                else:
                    pct = total_val * 100.0

                if pct >= 2.0:
                    ax.text(
                        x[i],
                        total_val * 1.01,
                        f"{pct:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                else:
                    ax.scatter(
                        x[i],
                        total_val,
                        color="red",
                        s=15,
                        zorder=4,
                    )

        axis_labels = [
            self._format_stoich_for_axis(lab) for lab in stoich_labels
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(axis_labels, rotation=45, ha="right")

        if weighting == "centers":
            base_label = f"{self.node_element} centers"
        elif weighting == "clusters":
            base_label = "clusters"
        else:
            base_label = "solute atoms"

        if y_mode == "count":
            ax.set_ylabel(f"Count of {base_label}")
        elif y_mode == "fraction":
            ax.set_ylabel(f"Fraction of {base_label}")
        else:
            ax.set_ylabel(f"Percentage of {base_label} [%]")

        if weighting == "centers":
            weight_desc = f"{self.node_element}-center-weighted"
        elif weighting == "clusters":
            weight_desc = "cluster-weighted"
        else:
            weight_desc = "solute-atom-weighted"

        ax.set_xlabel(f"{self.node_element}:{self.anion_element} structure")
        ax.set_title(
            f"Structure-Fraction Prior Histogram\n({weight_desc}, y={y_mode})"
        )

        ax.legend(
            title=f"{solvent_label} coordination",
            fontsize="small",
            title_fontsize="small",
        )
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        if show:
            plt.show()

        return dist_df, ax

    def export_distribution_txt(
        self,
        weighting: Literal["centers", "clusters", "solute_atoms"] = "centers",
        y_mode: Literal["count", "fraction", "percent"] = "fraction",
        solvent_label: str = "solvent",
        filename: Optional[str] = None,
        split_files: bool = False,
        fmt: Literal["blocks", "flat"] = "blocks",
        stoich_totals_filename: Optional[str] = None,
    ) -> List[str]:
        """Export the distribution information to .txt.

        Adds per-stoichiometry totals:
        - raw_cluster_count: number of clusters for that stoich (unweighted)
        - total_weight: sum of weights across n_solv_donors for that stoich
        - total_value: sum of values across n_solv_donors for that stoich

        Also writes a secondary file with 4 columns:
        stoich_label, order_index (0-based), total_counts, total_percent

        total_counts  = total_weight (so it matches your chosen weighting)
        total_percent = 100 * total_weight / (sum of total_weight across all stoich)

        Returns list of file paths written (includes the secondary totals file)
        """
        dist_df = self.compute_distribution(weighting=weighting, y_mode=y_mode)

        pivot = dist_df.pivot_table(
            index="stoich_label",
            columns="n_solv_donors",
            values="value",
            fill_value=0.0,
        )
        order_df = self._pb_i_sort_order()
        common_labels = [lab for lab in order_df.index if lab in pivot.index]
        sorted_labels = sorted(
            common_labels, key=lambda lab: order_df.loc[lab, "sort_key"]
        )

        stoich_meta = (
            self.df[
                [
                    "stoich_label",
                    "node_count",
                    "anion_count",
                    "solute_atom_count",
                ]
            ]
            .drop_duplicates(subset=["stoich_label"])
            .set_index("stoich_label")
        )

        def _safe_name(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.]+", "_", str(s)).strip("_")

        def _fmt_weight(v: float) -> str:
            if y_mode == "count":
                return f"{v:.0f}"
            return f"{v:.8f}"

        def _fmt_value(v: float) -> str:
            if y_mode == "count":
                return f"{v:.0f}"
            if y_mode == "percent":
                return f"{v:.6f}"
            return f"{v:.8f}"

        def _fmt_counts(v: float) -> str:
            # for totals file "total_counts"
            if abs(v - round(v)) < 1e-9:
                return f"{int(round(v))}"
            return f"{v:.8f}"

        def _fmt_percent(v: float) -> str:
            return f"{v:.6f}"

        header_lines = [
            "ClusterHistogramAnalyzer distribution export",
            f"metadata_path: {self.metadata_path}",
            f"node_element: {self.node_element}",
            f"anion_element: {self.anion_element}",
            f"weighting: {weighting}",
            f"y_mode: {y_mode}",
            f"solvent_label: {solvent_label}",
            "",
        ]

        def _per_stoich_totals(label: str) -> dict:
            sub = dist_df[dist_df["stoich_label"] == label]
            total_weight = float(sub["weight"].sum()) if len(sub) else 0.0
            total_value = float(sub["value"].sum()) if len(sub) else 0.0
            raw_cluster_count = int((self.df["stoich_label"] == label).sum())
            return {
                "raw_cluster_count": raw_cluster_count,
                "total_weight": total_weight,
                "total_value": total_value,
            }

        # Precompute totals table in the plot order
        totals_rows = []
        for idx, lab in enumerate(sorted_labels):
            t = _per_stoich_totals(lab)
            totals_rows.append(
                {
                    "stoich_label": lab,
                    "order_index": idx,
                    "raw_cluster_count": t["raw_cluster_count"],
                    "total_weight": t["total_weight"],
                    "total_value": t["total_value"],
                }
            )

        totals_df = pd.DataFrame(totals_rows)
        grand_total_weight = (
            float(totals_df["total_weight"].sum()) if len(totals_df) else 0.0
        )
        if grand_total_weight > 0:
            totals_df["total_percent"] = (
                100.0 * totals_df["total_weight"] / grand_total_weight
            )
        else:
            totals_df["total_percent"] = 0.0

        out_paths: List[str] = []

        # Decide naming base for the secondary totals file
        if filename is None:
            base_name = f"histogram_distribution_{weighting}_{y_mode}"
        else:
            base_name = os.path.splitext(os.path.basename(filename))[0]

        if stoich_totals_filename is None:
            stoich_totals_filename = f"{base_name}_stoich_totals.txt"
        if not stoich_totals_filename.lower().endswith(".txt"):
            stoich_totals_filename = stoich_totals_filename + ".txt"
        stoich_totals_path = os.path.join(
            self.output_dir, stoich_totals_filename
        )

        def _write_stoich_totals_file() -> None:
            # EXACTLY 4 columns as requested
            with open(stoich_totals_path, "w", encoding="utf-8") as f:
                f.write(
                    "stoich_label\torder_index\ttotal_counts\ttotal_percent\n"
                )
                for _, row in totals_df.iterrows():
                    f.write(
                        f"{row['stoich_label']}\t{int(row['order_index'])}\t"
                        f"{_fmt_counts(float(row['total_weight']))}\t{_fmt_percent(float(row['total_percent']))}\n"
                    )

        if split_files:
            for lab in sorted_labels:
                sub = dist_df[dist_df["stoich_label"] == lab].copy()
                sub = sub.sort_values("n_solv_donors")

                meta = (
                    stoich_meta.loc[lab] if lab in stoich_meta.index else None
                )
                axis_label = self._format_stoich_for_axis(lab)
                t = _per_stoich_totals(lab)

                if filename is None:
                    out_name = f"histogram_distribution_{_safe_name(lab)}_{weighting}_{y_mode}.txt"
                else:
                    base, ext = os.path.splitext(filename)
                    if not ext:
                        ext = ".txt"
                    out_name = f"{base}_{_safe_name(lab)}{ext}"

                out_path = os.path.join(self.output_dir, out_name)

                with open(out_path, "w", encoding="utf-8") as f:
                    for line in header_lines:
                        f.write(line + "\n")

                    f.write(f"stoich_label: {lab}\n")
                    f.write(f"axis_label: {axis_label}\n")
                    if meta is not None:
                        f.write(f"node_count: {int(meta['node_count'])}\n")
                        f.write(f"anion_count: {int(meta['anion_count'])}\n")
                        f.write(
                            f"solute_atom_count: {int(meta['solute_atom_count'])}\n"
                        )

                    f.write(f"raw_cluster_count: {t['raw_cluster_count']}\n")
                    f.write(
                        f"total_weight: {_fmt_weight(t['total_weight'])}\n"
                    )
                    f.write(f"total_value: {_fmt_value(t['total_value'])}\n")
                    f.write("\n")

                    if fmt == "flat":
                        f.write("stoich_label\tn_solv_donors\tweight\tvalue\n")
                        for _, row in sub.iterrows():
                            f.write(
                                f"{row['stoich_label']}\t{int(row['n_solv_donors'])}\t"
                                f"{_fmt_weight(float(row['weight']))}\t{_fmt_value(float(row['value']))}\n"
                            )
                    else:
                        f.write("n_solv_donors\tweight\tvalue\n")
                        for _, row in sub.iterrows():
                            f.write(
                                f"{int(row['n_solv_donors'])}\t"
                                f"{_fmt_weight(float(row['weight']))}\t{_fmt_value(float(row['value']))}\n"
                            )

                out_paths.append(out_path)

            _write_stoich_totals_file()
            out_paths.append(stoich_totals_path)
            return out_paths

        # Single combined file
        if filename is None:
            filename = f"{base_name}.txt"
        if not filename.lower().endswith(".txt"):
            filename = filename + ".txt"
        out_path = os.path.join(self.output_dir, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")

            if fmt == "flat":
                f.write("stoich_label\tn_solv_donors\tweight\tvalue\n")
                for lab in sorted_labels:
                    sub = dist_df[dist_df["stoich_label"] == lab].copy()
                    sub = sub.sort_values("n_solv_donors")
                    for _, row in sub.iterrows():
                        f.write(
                            f"{row['stoich_label']}\t{int(row['n_solv_donors'])}\t"
                            f"{_fmt_weight(float(row['weight']))}\t{_fmt_value(float(row['value']))}\n"
                        )

                f.write("\n")
                f.write("== SUMMARY ==\n")
                f.write(
                    "stoich_label\torder_index\traw_cluster_count\ttotal_weight\ttotal_value\ttotal_percent\n"
                )
                for _, row in totals_df.iterrows():
                    f.write(
                        f"{row['stoich_label']}\t{int(row['order_index'])}\t{int(row['raw_cluster_count'])}\t"
                        f"{_fmt_weight(float(row['total_weight']))}\t{_fmt_value(float(row['total_value']))}\t"
                        f"{_fmt_percent(float(row['total_percent']))}\n"
                    )

            else:
                for lab in sorted_labels:
                    sub = dist_df[dist_df["stoich_label"] == lab].copy()
                    sub = sub.sort_values("n_solv_donors")

                    meta = (
                        stoich_meta.loc[lab]
                        if lab in stoich_meta.index
                        else None
                    )
                    axis_label = self._format_stoich_for_axis(lab)
                    t = _per_stoich_totals(lab)

                    f.write(f"== {lab} ==\n")
                    f.write(f"axis_label: {axis_label}\n")
                    if meta is not None:
                        f.write(f"node_count: {int(meta['node_count'])}\n")
                        f.write(f"anion_count: {int(meta['anion_count'])}\n")
                        f.write(
                            f"solute_atom_count: {int(meta['solute_atom_count'])}\n"
                        )

                    f.write(f"raw_cluster_count: {t['raw_cluster_count']}\n")
                    f.write(
                        f"total_weight: {_fmt_weight(t['total_weight'])}\n"
                    )
                    f.write(f"total_value: {_fmt_value(t['total_value'])}\n")

                    f.write("n_solv_donors\tweight\tvalue\n")
                    for _, row in sub.iterrows():
                        f.write(
                            f"{int(row['n_solv_donors'])}\t"
                            f"{_fmt_weight(float(row['weight']))}\t{_fmt_value(float(row['value']))}\n"
                        )
                    f.write("\n")

                f.write("== SUMMARY ==\n")
                f.write(
                    "stoich_label\torder_index\traw_cluster_count\ttotal_weight\ttotal_value\ttotal_percent\n"
                )
                for _, row in totals_df.iterrows():
                    f.write(
                        f"{row['stoich_label']}\t{int(row['order_index'])}\t{int(row['raw_cluster_count'])}\t"
                        f"{_fmt_weight(float(row['total_weight']))}\t{_fmt_value(float(row['total_value']))}\t"
                        f"{_fmt_percent(float(row['total_percent']))}\n"
                    )

        out_paths.append(out_path)

        _write_stoich_totals_file()
        out_paths.append(stoich_totals_path)

        return out_paths


# @dataclass
# class XYZClusterConfig:
#     """
#     Configuration for XYZ-based solute cluster extraction.

#     Parameters
#     ----------
#     node_elements : set of str
#         Elements that act as "nodes" (e.g. {"Pb"}).
#     linker_elements : set of str
#         Elements that act as "linkers" (e.g. {"I"}).
#     connectivity_cutoffs : dict[(str,str) -> float]
#         Cutoffs for building the solute connectivity graph.
#         Example: {("Pb","I"): 3.5}.
#     coordination_cutoffs : dict[(str,str) -> float]
#         Cutoffs for coordination counting.
#         Example: {("Pb","O"): 3.0}.
#     shell_radius : float
#         Radius (Å) for spherical extraction around node centers.
#     box : array-like of length 3, optional
#         Simulation box lengths [Lx, Ly, Lz] for PBC. If None, no PBC.
#     solute_elements : set of str, optional
#         Elements considered part of the solute (defaults to node ∪ linker).
#     solvent_donor_element : str, optional
#         Element used to classify number of coordinated solvents, e.g. "O".
#         For DMF-like systems, you can treat each coordinated O as one solvent.
#     """
#     node_elements: Set[str]
#     linker_elements: Set[str]
#     connectivity_cutoffs: Dict[Tuple[str, str], float]
#     coordination_cutoffs: Dict[Tuple[str, str], float]
#     shell_radius: float
#     box: Optional[np.ndarray] = None
#     solute_elements: Optional[Set[str]] = None
#     solvent_donor_element: Optional[str] = None

#     def __post_init__(self) -> None:
#         # Normalize element symbols to Title case ("Pb", "I", "O")
#         self.node_elements = {e.title() for e in self.node_elements}
#         self.linker_elements = {e.title() for e in self.linker_elements}
#         if self.solute_elements is None:
#             self.solute_elements = set(self.node_elements) | set(self.linker_elements)
#         else:
#             self.solute_elements = {e.title() for e in self.solute_elements}

#         # Normalize coordination + connectivity keys to Title case
#         self.connectivity_cutoffs = {
#             (a.title(), b.title()): float(r)
#             for (a, b), r in self.connectivity_cutoffs.items()
#         }
#         self.coordination_cutoffs = {
#             (a.title(), b.title()): float(r)
#             for (a, b), r in self.coordination_cutoffs.items()
#         }

#         if self.box is not None:
#             self.box = np.asarray(self.box, dtype=float)

#         if self.solvent_donor_element is not None:
#             self.solvent_donor_element = self.solvent_donor_element.title()

#         if self.shell_radius <= 0:
#             raise ValueError("shell_radius must be positive")

#     @property
#     def max_connectivity_cutoff(self) -> float:
#         if not self.connectivity_cutoffs:
#             return 0.0
#         return max(self.connectivity_cutoffs.values())

#     @property
#     def max_coordination_cutoff(self) -> float:
#         if not self.coordination_cutoffs:
#             return 0.0
#         return max(self.coordination_cutoffs.values())


# @dataclass
# class XYZFrame:
#     """
#     A single XYZ frame.

#     Attributes
#     ----------
#     symbols : (N,) array of str
#         Element symbols for each atom.
#     coords : (N,3) ndarray
#         Cartesian coordinates in Å (in simulation box).
#     box : (3,) ndarray or None
#         Box lengths [Lx, Ly, Lz]. If None, no PBC is applied.
#     frame_id : str
#         Identifier (e.g. file stem or "frame_0001").
#     file_path : str
#         Source XYZ file path for this frame.
#     """
#     symbols: np.ndarray
#     coords: np.ndarray
#     box: Optional[np.ndarray]
#     frame_id: str
#     file_path: str


# @dataclass
# class SoluteCluster:
#     """
#     A connected solute cluster (e.g. Pb2I4).

#     Attributes
#     ----------
#     frame_id : str
#     cluster_index : int
#         Index local to the frame (1-based).
#     atom_indices : list of int
#         Indices of solute atoms in this cluster.
#     node_indices : list of int
#         Subset of atom_indices that are "nodes" (e.g. Pb).
#     stoich_counts : dict[str,int]
#         Element -> count for solute atoms in this cluster.
#     stoich_label : str
#         Formula-like label (e.g. "Pb2I4").
#     virtual_positions : dict[int, np.ndarray]
#         Virtual coordinates for atoms in this cluster's BFS space.
#         - For solute atoms, set during connectivity BFS.
#         - For shell atoms, set during shell extraction.
#     """
#     frame_id: str
#     cluster_index: int
#     atom_indices: List[int]
#     node_indices: List[int]
#     stoich_counts: Dict[str, int]
#     stoich_label: str
#     virtual_positions: Dict[int, np.ndarray] = field(default_factory=dict)


# # ---------------------------------------------------------------------------
# # Main extractor
# # ---------------------------------------------------------------------------

# class XYZClusterExtractor:
#     """
#     Extracts solute clusters + spherical solvent shells from a folder of XYZ files.

#     Types assumed to exist:
#     - XYZClusterConfig
#     - XYZFrame
#     - SoluteCluster
#     """

#     def __init__(
#         self,
#         input_dir: str,
#         output_dir: str,
#         config: XYZClusterConfig,
#         box_override: Optional[np.ndarray] = None,
#         use_pbc: bool = False,
#         debug_pbc: bool = False,
#     ) -> None:
#         """
#         Parameters
#         ----------
#         input_dir : str
#             Folder containing XYZ files (one frame per file).
#         output_dir : str
#             Folder to write cluster XYZ files + metadata CSV.
#         config : XYZClusterConfig
#             Configuration object describing solute/solvent and cutoffs.
#         box_override : array-like of length 3, optional
#             If given, overrides any per-frame box and config.box.
#         use_pbc : bool, optional
#             If True, use periodic boundary conditions with the given box.
#             If False, ignore any box information and work in open space and
#             do NOT modify coordinates in output (aside from subsetting atoms).
#         debug_pbc : bool, optional
#             If True, print detailed information about atoms that required PBC
#             wrapping: raw box coords, virtual coords, and output coords.
#         """
#         self.input_dir = os.path.abspath(input_dir)
#         self.output_dir = os.path.abspath(output_dir)
#         self.config = config
#         self.box_override = np.asarray(box_override, dtype=float) if box_override is not None else None
#         self.use_pbc = bool(use_pbc)
#         self.debug_pbc = bool(debug_pbc)

#         os.makedirs(self.output_dir, exist_ok=True)

#         self._cluster_ids_used: Set[str] = set()
#         self._metadata_rows: List[Dict[str, object]] = []

#     # ---------------------------- Public API -----------------------------

#     def run(self) -> None:
#         """
#         Process all XYZ files in input_dir and write cluster XYZs + metadata CSV.
#         """
#         xyz_files = sorted(
#             f for f in os.listdir(self.input_dir)
#             if f.lower().endswith(".xyz")
#         )

#         global_cluster_counter = 0

#         for frame_idx, filename in enumerate(xyz_files, start=1):
#             file_path = os.path.join(self.input_dir, filename)
#             frame = self._load_xyz_file(file_path, frame_idx)

#             kd_tree = KDTree(frame.coords)
#             clusters = self._find_solute_clusters(frame, kd_tree)

#             for cluster in clusters:
#                 global_cluster_counter += 1

#                 extracted_indices = self._extract_shell_for_cluster(
#                     frame=frame,
#                     kd_tree=kd_tree,
#                     cluster=cluster
#                 )

#                 coord_per_center, total_coord_counts = self._compute_coordination_for_cluster(
#                     frame=frame,
#                     kd_tree=kd_tree,
#                     cluster=cluster,
#                     extracted_indices=extracted_indices
#                 )

#                 if self.config.solvent_donor_element is not None:
#                     donor = self.config.solvent_donor_element
#                     n_solv_donors = int(total_coord_counts.get(donor, 0))
#                 else:
#                     n_solv_donors = 0

#                 xyz_path = self._write_cluster_xyz(
#                     frame=frame,
#                     cluster=cluster,
#                     extracted_indices=extracted_indices,
#                     n_solv_donors=n_solv_donors
#                 )

#                 meta = self._build_metadata_row(
#                     frame=frame,
#                     cluster=cluster,
#                     extracted_indices=extracted_indices,
#                     total_coord_counts=total_coord_counts,
#                     coord_per_center=coord_per_center,
#                     n_solv_donors=n_solv_donors,
#                     xyz_path=xyz_path,
#                     frame_index=frame_idx,
#                     global_cluster_index=global_cluster_counter
#                 )
#                 self._metadata_rows.append(meta)

#         if self._metadata_rows:
#             self._write_metadata_csv()

#     # ---------------------- XYZ loading / frame box ----------------------

#     def _load_xyz_file(self, file_path: str, frame_idx: int) -> XYZFrame:
#         """
#         Load a standard XYZ file: first line N, second line comment, then N rows of:
#             symbol x y z
#         """
#         with open(file_path, "r") as f:
#             lines = [line.strip() for line in f if line.strip()]

#         try:
#             n_atoms = int(lines[0].split()[0])
#         except Exception as exc:
#             raise ValueError(f"Failed to parse atom count from first line of {file_path}") from exc

#         if len(lines) < n_atoms + 2:
#             raise ValueError(f"XYZ file {file_path} does not have enough lines for {n_atoms} atoms")

#         atom_lines = lines[2:2 + n_atoms]

#         symbols: List[str] = []
#         coords: List[List[float]] = []

#         for line in atom_lines:
#             parts = line.split()
#             if len(parts) < 4:
#                 raise ValueError(f"Malformed atom line in {file_path}: '{line}'")
#             symbols.append(parts[0].title())
#             x, y, z = map(float, parts[1:4])
#             coords.append([x, y, z])

#         symbols_arr = np.array(symbols, dtype=object)
#         coords_arr = np.array(coords, dtype=float)

#         if self.box_override is not None:
#             box = self.box_override
#         else:
#             box = self.config.box

#         frame_id = os.path.splitext(os.path.basename(file_path))[0]

#         return XYZFrame(
#             symbols=symbols_arr,
#             coords=coords_arr,
#             box=box,
#             frame_id=frame_id,
#             file_path=file_path
#         )

#     def _get_box(self, frame: XYZFrame) -> Optional[np.ndarray]:
#         """
#         Return the box vector to use for a given frame.

#         If use_pbc is False, always returns None (PBC disabled).
#         Otherwise, priority: box_override > frame.box > config.box.
#         """
#         if not self.use_pbc:
#             return None

#         if self.box_override is not None:
#             return self.box_override
#         if frame.box is not None:
#             return frame.box
#         return self.config.box

#     # -------------------------- PBC utilities ----------------------------

#     @staticmethod
#     def _pbc_distance(
#         pos1: np.ndarray,
#         pos2: np.ndarray,
#         box: Optional[np.ndarray]
#     ) -> float:
#         """
#         Minimum-image distance between pos1 and pos2 given box.
#         If box is None, returns straight Euclidean distance.
#         """
#         delta = pos2 - pos1
#         if box is not None:
#             delta = delta - np.round(delta / box) * box
#         return float(np.linalg.norm(delta))

#     def _neighbors_within_cutoff(
#         self,
#         frame: XYZFrame,
#         kd_tree: KDTree,
#         parent_index: int,
#         cutoff: float
#     ) -> List[int]:
#         """
#         Returns indices of atoms within cutoff of parent_index.

#         If PBC is disabled (use_pbc=False) or no box is available, this uses a
#         simple KDTree query_ball_point in Euclidean space.

#         If PBC is enabled and a box is available, it generates image positions
#         of the parent in neighboring cells and filters neighbors using the
#         minimum-image distance.
#         """
#         coords = frame.coords
#         box = self._get_box(frame)
#         parent_pos = coords[parent_index]
#         neighbors: Set[int] = set()

#         if cutoff <= 0:
#             return []

#         # No PBC: straight Euclidean query
#         if box is None:
#             candidate_idx = kd_tree.query_ball_point(parent_pos, r=cutoff)
#             for idx in candidate_idx:
#                 if idx != parent_index:
#                     neighbors.add(idx)
#             return list(neighbors)

#         # With PBC: generate image positions in neighboring cells
#         shifts = (-1, 0, 1)
#         for dx in shifts:
#             for dy in shifts:
#                 for dz in shifts:
#                     image_pos = parent_pos + np.array(
#                         [dx * box[0], dy * box[1], dz * box[2]],
#                         dtype=float
#                     )
#                     candidate_idx = kd_tree.query_ball_point(image_pos, r=cutoff)
#                     for idx in candidate_idx:
#                         if idx == parent_index:
#                             continue
#                         pos = coords[idx]
#                         dist = self._pbc_distance(parent_pos, pos, box)
#                         if dist <= cutoff:
#                             neighbors.add(idx)

#         return list(neighbors)

#     # ---------------------- Cutoff helpers -------------------------------

#     def _get_connectivity_cutoff(
#         self,
#         elem_a: str,
#         elem_b: str
#     ) -> Optional[float]:
#         dct = self.config.connectivity_cutoffs
#         if (elem_a, elem_b) in dct:
#             return dct[(elem_a, elem_b)]
#         if (elem_b, elem_a) in dct:
#             return dct[(elem_b, elem_a)]
#         return None

#     def _get_coordination_cutoff(
#         self,
#         elem_center: str,
#         elem_neighbor: str
#     ) -> Optional[float]:
#         dct = self.config.coordination_cutoffs
#         if (elem_center, elem_neighbor) in dct:
#             return dct[(elem_center, elem_neighbor)]
#         if (elem_neighbor, elem_center) in dct:
#             return dct[(elem_neighbor, elem_center)]
#         return None

#     # ---------------------- Solute clustering logic ----------------------

#     def _solute_connectivity_neighbors(
#         self,
#         frame: XYZFrame,
#         kd_tree: KDTree,
#         parent_index: int
#     ) -> List[int]:
#         """
#         Returns neighbors of parent_index that:
#         - are solute elements
#         - are within the appropriate connectivity cutoff for (elem_parent, elem_neighbor)
#           (PBC optional).
#         """
#         coords = frame.coords
#         symbols = frame.symbols

#         max_cutoff = self.config.max_connectivity_cutoff
#         if max_cutoff <= 0:
#             return []

#         elem_parent = symbols[parent_index].title()
#         candidate_indices = self._neighbors_within_cutoff(
#             frame=frame,
#             kd_tree=kd_tree,
#             parent_index=parent_index,
#             cutoff=max_cutoff
#         )

#         box = self._get_box(frame)

#         neighbors: List[int] = []
#         for idx in candidate_indices:
#             elem_neighbor = symbols[idx].title()
#             if elem_neighbor not in self.config.solute_elements:
#                 continue

#             cutoff = self._get_connectivity_cutoff(elem_parent, elem_neighbor)
#             if cutoff is None:
#                 continue

#             dist = self._pbc_distance(coords[parent_index], coords[idx], box)
#             if dist <= cutoff:
#                 neighbors.append(idx)

#         return neighbors

#     def _find_solute_clusters(
#         self,
#         frame: XYZFrame,
#         kd_tree: KDTree
#     ) -> List[SoluteCluster]:
#         """
#         Find all connected solute clusters (solute elements only) in a frame.

#         Also constructs a consistent set of virtual positions for the solute
#         atoms in each cluster, using BFS:
#         - The first atom in the component sets the origin.
#         - Each neighbor is placed at parent_virtual + min-image(delta).

#         If debug_pbc is True, prints information for any solute atom whose
#         virtual position required a non-zero PBC shift relative to its raw
#         box coordinate.
#         """
#         coords = frame.coords
#         symbols = frame.symbols
#         box = self._get_box(frame)
#         solute_elements = self.config.solute_elements

#         solute_indices = [
#             i for i, elem in enumerate(symbols)
#             if elem in solute_elements
#         ]

#         visited: Set[int] = set()
#         clusters: List[SoluteCluster] = []
#         cluster_local_index = 0

#         for idx in solute_indices:
#             if idx in visited:
#                 continue

#             cluster_local_index += 1
#             queue: deque[int] = deque([idx])
#             visited.add(idx)
#             cluster_atoms: Set[int] = set([idx])

#             virt: Dict[int, np.ndarray] = {}
#             virt[idx] = coords[idx].copy()

#             while queue:
#                 current = queue.popleft()
#                 neighbors = self._solute_connectivity_neighbors(frame, kd_tree, current)
#                 for nb in neighbors:
#                     if nb in visited:
#                         continue
#                     visited.add(nb)
#                     cluster_atoms.add(nb)
#                     queue.append(nb)

#                     raw_delta = coords[nb] - coords[current]
#                     if box is not None:
#                         shift_vec = np.round(raw_delta / box).astype(int)
#                         delta = raw_delta - shift_vec * box
#                     else:
#                         shift_vec = np.zeros(3, dtype=int)
#                         delta = raw_delta

#                     virt[nb] = virt[current] + delta

#                     if self.debug_pbc and box is not None and np.any(shift_vec != 0):
#                         print(
#                             "[PBC-BFS] frame="
#                             f"{frame.frame_id}, cluster_index={cluster_local_index}, "
#                             f"solute_atom={nb}, elem={symbols[nb]} "
#                             f"parent={current}, shift={tuple(shift_vec)}, "
#                             f"raw={coords[nb]}, parent_raw={coords[current]}, "
#                             f"virt={virt[nb]}"
#                         )

#             atom_indices = sorted(cluster_atoms)
#             node_indices = [
#                 i for i in atom_indices
#                 if symbols[i] in self.config.node_elements
#             ]
#             if not node_indices:
#                 node_indices = atom_indices.copy()

#             stoich_counts = Counter(symbols[i] for i in atom_indices)
#             stoich_label = self._stoichiometry_label(stoich_counts)

#             cluster_virt = {i: virt[i] for i in atom_indices}

#             clusters.append(
#                 SoluteCluster(
#                     frame_id=frame.frame_id,
#                     cluster_index=cluster_local_index,
#                     atom_indices=atom_indices,
#                     node_indices=node_indices,
#                     stoich_counts=dict(stoich_counts),
#                     stoich_label=stoich_label,
#                     virtual_positions=cluster_virt
#                 )
#             )

#         return clusters

#     @staticmethod
#     def _stoichiometry_label(stoich_counts: Dict[str, int]) -> str:
#         """
#         Convert a dict {'Pb': 2, 'I': 4} to a label like "Pb2I4".
#         """
#         parts: List[str] = []
#         for elem in sorted(stoich_counts.keys()):
#             count = stoich_counts[elem]
#             if count == 1:
#                 parts.append(elem)
#             else:
#                 parts.append(f"{elem}{count}")
#         return "".join(parts)

#     # ---------------------- Shell extraction logic -----------------------

#     def _extract_shell_for_cluster(
#         self,
#         frame: XYZFrame,
#         kd_tree: KDTree,
#         cluster: SoluteCluster
#     ) -> List[int]:
#         """
#         For a given solute cluster, extract all atoms within shell_radius of each node,
#         excluding solute atoms not in this cluster.

#         Also assigns virtual positions to shell atoms when PBC is active, using:
#             virt_shell = virt_center + min-image(coords_shell - coords_center)
#         so the whole cluster (solute + shell) lives in a consistent virtual space.

#         If debug_pbc is True, prints information for any shell atom whose virtual
#         position required a non-zero PBC shift relative to its raw box coordinate.
#         """
#         coords = frame.coords
#         symbols = frame.symbols
#         box = self._get_box(frame)

#         solute_elements = self.config.solute_elements
#         shell_radius = self.config.shell_radius

#         solute_set: Set[int] = set(cluster.atom_indices)
#         extracted: Set[int] = set(cluster.atom_indices)

#         virt = cluster.virtual_positions

#         if shell_radius <= 0:
#             return sorted(extracted)

#         for center_idx in cluster.node_indices:
#             center_virt = virt.get(center_idx, coords[center_idx])
#             candidate_indices = self._neighbors_within_cutoff(
#                 frame=frame,
#                 kd_tree=kd_tree,
#                 parent_index=center_idx,
#                 cutoff=shell_radius
#             )
#             for idx in candidate_indices:
#                 elem = symbols[idx].title()
#                 if elem in solute_elements and idx not in solute_set:
#                     continue

#                 if idx not in extracted:
#                     extracted.add(idx)

#                     raw_delta = coords[idx] - coords[center_idx]
#                     if box is not None:
#                         shift_vec = np.round(raw_delta / box).astype(int)
#                         delta = raw_delta - shift_vec * box
#                     else:
#                         shift_vec = np.zeros(3, dtype=int)
#                         delta = raw_delta

#                     if idx not in virt:
#                         virt[idx] = center_virt + delta

#                     if self.debug_pbc and box is not None and np.any(shift_vec != 0):
#                         print(
#                             "[PBC-SHELL] frame="
#                             f"{frame.frame_id}, cluster_index={cluster.cluster_index}, "
#                             f"shell_atom={idx}, elem={symbols[idx]} "
#                             f"center={center_idx}, shift={tuple(shift_vec)}, "
#                             f"raw={coords[idx]}, center_raw={coords[center_idx]}, "
#                             f"virt={virt[idx]}"
#                         )

#         return sorted(extracted)

#     # -------------------- Coordination number analysis -------------------

#     def _compute_coordination_for_cluster(
#         self,
#         frame: XYZFrame,
#         kd_tree: KDTree,
#         cluster: SoluteCluster,
#         extracted_indices: List[int]
#     ) -> Tuple[Dict[int, Dict[str, int]], Counter]:
#         """
#         Compute coordination numbers for each node in the cluster.
#         """
#         coords = frame.coords
#         symbols = frame.symbols
#         box = self._get_box(frame)

#         extracted_set: Set[int] = set(extracted_indices)
#         max_cutoff = self.config.max_coordination_cutoff

#         coord_per_center: Dict[int, Dict[str, int]] = {}
#         total_counts: Counter = Counter()

#         if max_cutoff <= 0:
#             return coord_per_center, total_counts

#         for center_idx in cluster.node_indices:
#             elem_center = symbols[center_idx].title()
#             center_counts: Counter = Counter()

#             candidate_indices = self._neighbors_within_cutoff(
#                 frame=frame,
#                 kd_tree=kd_tree,
#                 parent_index=center_idx,
#                 cutoff=max_cutoff
#             )

#             for idx in candidate_indices:
#                 if idx == center_idx:
#                     continue
#                 if idx not in extracted_set:
#                     continue

#                 elem_neighbor = symbols[idx].title()
#                 cutoff = self._get_coordination_cutoff(elem_center, elem_neighbor)
#                 if cutoff is None:
#                     continue

#                 dist = self._pbc_distance(coords[center_idx], coords[idx], box)
#                 if dist <= cutoff:
#                     center_counts[elem_neighbor] += 1

#             coord_per_center[center_idx] = dict(center_counts)
#             total_counts.update(center_counts)

#         return coord_per_center, total_counts

#     # -------------------------- XYZ writing ------------------------------

#     def _write_cluster_xyz(
#         self,
#         frame: XYZFrame,
#         cluster: SoluteCluster,
#         extracted_indices: List[int],
#         n_solv_donors: int
#     ) -> str:
#         """
#         Write an XYZ file for a cluster + shell.

#         Atoms are written in an order where solute atoms come first,
#         then solvent atoms.

#         Behaviour:
#         - If use_pbc is False: coordinates are copied exactly from the input
#           XYZ (no translation, no wrapping), just a subset of atoms.
#         - If use_pbc is True: coordinates come from the cluster's virtual
#           positions and are translated so the reference atom sits at the origin.
#         """
#         symbols = frame.symbols
#         coords = frame.coords

#         stoich_label = cluster.stoich_label
#         solv_folder_name = f"solv_{n_solv_donors}"

#         stoich_dir = os.path.join(self.output_dir, stoich_label)
#         solv_dir = os.path.join(stoich_dir, solv_folder_name)
#         os.makedirs(solv_dir, exist_ok=True)

#         solute_set = set(cluster.atom_indices)
#         extracted_set = set(extracted_indices)

#         solute_order = sorted(solute_set)
#         solvent_order = sorted(extracted_set - solute_set)
#         ordered_indices = solute_order + solvent_order

#         virt = cluster.virtual_positions

#         if cluster.node_indices:
#             ref_idx = cluster.node_indices[0]
#         else:
#             ref_idx = cluster.atom_indices[0]
#         ref_pos = virt.get(ref_idx, coords[ref_idx])

#         new_coords: List[np.ndarray] = []

#         if not self.use_pbc:
#             # No PBC: copy raw coordinates exactly
#             for idx in ordered_indices:
#                 raw_pos = coords[idx].copy()
#                 new_coords.append(raw_pos)

#                 if self.debug_pbc:
#                     print(
#                         "[OUT-noPBC] frame="
#                         f"{frame.frame_id}, cluster_index={cluster.cluster_index}, "
#                         f"atom={idx}, elem={symbols[idx]}, raw={raw_pos}"
#                     )
#         else:
#             # PBC mode: use virtual coordinates and translate by ref_pos
#             for idx in ordered_indices:
#                 raw_pos = coords[idx]
#                 virt_pos = virt.get(idx, raw_pos)
#                 out_pos = virt_pos - ref_pos
#                 new_coords.append(out_pos)

#                 if self.debug_pbc:
#                     wrapped_flag = np.any(np.abs(virt_pos - raw_pos) > 1e-6)
#                     print(
#                         "[PBC-OUT] frame="
#                         f"{frame.frame_id}, cluster_index={cluster.cluster_index}, "
#                         f"atom={idx}, elem={symbols[idx]}, "
#                         f"wrapped={wrapped_flag}, "
#                         f"raw={raw_pos}, virt={virt_pos}, out={out_pos}"
#                     )

#         file_name = (
#             f"{stoich_label}_frame{frame.frame_id}_"
#             f"cluster{cluster.cluster_index:03d}_"
#             f"solv{n_solv_donors}.xyz"
#         )
#         out_path = os.path.join(solv_dir, file_name)

#         with open(out_path, "w") as f:
#             f.write(f"{len(ordered_indices)}\n")
#             comment = (
#                 f"frame={frame.frame_id} "
#                 f"cluster_index={cluster.cluster_index} "
#                 f"stoich={stoich_label} "
#                 f"solv_donors={n_solv_donors} "
#                 f"pbc_mode={'on' if self.use_pbc else 'off'}\n"
#             )
#             f.write(comment)
#             for idx, pos in zip(ordered_indices, new_coords):
#                 elem = symbols[idx]
#                 x, y, z = pos
#                 f.write(f"{elem:2s} {x: .8f} {y: .8f} {z: .8f}\n")

#         return out_path

#     # ------------------------- Metadata bookkeeping ----------------------

#     @staticmethod
#     def _generate_cluster_id(
#         used_ids: Set[str]
#     ) -> str:
#         """
#         Generate a three-letter cluster ID (AAA..ZZZ) that is not used yet.
#         """
#         letters = ascii_uppercase
#         max_ids = 26 ** 3
#         n = 0
#         while n < max_ids:
#             temp = n
#             cid = ""
#             for _ in range(3):
#                 cid = letters[temp % 26] + cid
#                 temp //= 26
#             if cid not in used_ids:
#                 used_ids.add(cid)
#                 return cid
#             n += 1
#         raise ValueError("Exhausted all three-letter cluster IDs")

#     def _build_metadata_row(
#         self,
#         frame: XYZFrame,
#         cluster: SoluteCluster,
#         extracted_indices: List[int],
#         total_coord_counts: Counter,
#         coord_per_center: Dict[int, Dict[str, int]],
#         n_solv_donors: int,
#         xyz_path: str,
#         frame_index: int,
#         global_cluster_index: int
#     ) -> Dict[str, object]:
#         """
#         Build one metadata row for a cluster.
#         """
#         if total_coord_counts:
#             coord_summary = ";".join(
#                 f"{elem}:{count}"
#                 for elem, count in sorted(total_coord_counts.items())
#             )
#         else:
#             coord_summary = ""

#         per_center_parts: List[str] = []
#         for center_idx in sorted(coord_per_center.keys()):
#             counts = coord_per_center[center_idx]
#             if not counts:
#                 inner = "none"
#             else:
#                 inner = ",".join(f"{elem}={cnt}" for elem, cnt in sorted(counts.items()))
#             # Avoid "155:0=0" so Excel doesn't treat it like a time
#             per_center_parts.append(f"center{center_idx}[{inner}]")
#         per_center_summary = ";".join(per_center_parts)

#         cluster_id = self._generate_cluster_id(self._cluster_ids_used)

#         meta: Dict[str, object] = {
#             "global_cluster_index": global_cluster_index,
#             "cluster_id": cluster_id,
#             "frame_index": frame_index,
#             "frame_id": frame.frame_id,
#             "cluster_index_in_frame": cluster.cluster_index,
#             "stoich_label": cluster.stoich_label,
#             "n_atoms_solute": len(cluster.atom_indices),
#             "n_atoms_extracted": len(extracted_indices),
#             "n_solv_donors": n_solv_donors,
#             "coordination_totals": coord_summary,
#             "coordination_per_center": per_center_summary,
#             "xyz_path": os.path.relpath(xyz_path, start=self.output_dir),
#             "pbc_mode": "on" if self.use_pbc else "off",
#         }

#         for elem, count in sorted(cluster.stoich_counts.items()):
#             meta[f"solute_{elem}"] = count

#         return meta

#     def _write_metadata_csv(self) -> None:
#         """
#         Write the metadata rows to cluster_metadata.csv in output_dir.
#         """
#         if not self._metadata_rows:
#             return

#         out_path = os.path.join(self.output_dir, "cluster_metadata.csv")
#         fieldnames: List[str] = sorted(self._metadata_rows[0].keys())

#         with open(out_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for row in self._metadata_rows:
#                 writer.writerow(row)

#         print(f"Metadata written to: {out_path}")

# ---------------------------------------------------------------------------
