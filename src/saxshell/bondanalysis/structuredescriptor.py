import os
import re
import numpy as np
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from itertools import combinations
from datetime import datetime
from typing import List, Dict, Union, Tuple, Optional

@dataclass
class AtomPlus:
    atom_id: int
    atom_name: str
    residue_name: str
    residue_number: int
    x: float
    y: float
    z: float
    element: str

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Computes a 3×3 rotation matrix via Rodrigues' formula.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([[    0, -axis[2],  axis[1]],
                  [ axis[2],     0, -axis[0]],
                  [-axis[1],  axis[0],     0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

class StructureDescriptor:
    """
    Reads PDB/XYZ structures, aligns by anchor atoms, computes descriptors,
    geometric centers, coordination numbers, and provides visualization.

    Available descriptors after run():
      - self.histogram:        ndarray (n_structures × bins)
      - self.rmsd:             ndarray (n_structures,)
      - self.inertia:          ndarray (n_structures × 3)
      - self.element_centers:  List[Dict[str, ndarray]]
      - self.center_dists:     ndarray (n_structures × n_element_types)
      - self.coordination_numbers: List[Dict[Tuple[str,str], ndarray]]
    """
    def __init__(
        self,
        input_folder: str,
        anchor_atom: str = 'Pb',
        coord_cutoffs: Optional[Dict[Tuple[str,str], float]] = None
    ):  
        self.input_folder       = input_folder
        self.anchor_atom        = anchor_atom
        self.coord_cutoffs     = coord_cutoffs or {}

        # storage
        self.structures            : List[List[AtomPlus]] = []
        self.filenames             : List[str]           = []
        self.histogram             : np.ndarray          = None
        self.rmsd                  : np.ndarray          = None
        self.inertia               : np.ndarray          = None
        self.element_centers       : List[Dict[str, np.ndarray]] = []
        self.element_types         : List[str]           = []
        self.center_dists          : np.ndarray          = None
            # coordination numbers:
        self.coordination_numbers: List[Dict[Tuple[str,str], List[int]]] = []
        self.coordination_sets:    List[Dict[Tuple[str,str], Tuple[int,...]]] = []

    def read_structures(self) -> List[str]:
        """Read all .pdb/.xyz files in input_folder."""
        self.structures.clear()
        self.filenames .clear()
        files = sorted(f for f in os.listdir(self.input_folder)
                       if f.lower().endswith(('.pdb', '.xyz')))
        if not files:
            raise ValueError(f"No .pdb/.xyz files found in {self.input_folder!r}")
        for fn in files:
            path = os.path.join(self.input_folder, fn)
            if fn.lower().endswith('.pdb'):
                struct = self._read_pdb(path)
            else:
                struct = self._read_xyz(path)
            self.structures.append(struct)
            self.filenames.append(fn)
        return self.filenames

    def _read_pdb(self, filepath: str) -> List[AtomPlus]:
        atoms = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith(('ATOM','HETATM')):
                    atoms.append(self._parse_pdb_line(line))
        return atoms

    def _parse_pdb_line(self, line: str) -> AtomPlus:
        atom_id        = int(line[6:11].strip())
        atom_name      = line[12:16].strip()
        residue_name   = line[17:20].strip()
        residue_number = int(line[22:26].strip())
        x              = float(line[30:38].strip())
        y              = float(line[38:46].strip())
        z              = float(line[46:54].strip())
        element        = line[76:78].strip() or re.sub(r'[^A-Za-z]', '', atom_name)[:2]
        return AtomPlus(atom_id, atom_name, residue_name, residue_number, x, y, z, element)

    def _read_xyz(self, filepath: str) -> List[AtomPlus]:
        atoms = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        try:
            n = int(lines[0].strip())
        except ValueError:
            raise ValueError(f"XYZ file must start with atom count: {filepath}")
        for i, line in enumerate(lines[2:2+n], start=1):
            parts = line.split()
            elem, x, y, z = parts[0], *map(float, parts[1:4])
            atoms.append(AtomPlus(i, elem, '', 0, x, y, z, elem))
        return atoms

    def get_anchor_atoms(self, struct: List[AtomPlus]) -> List[AtomPlus]:
        return [a for a in struct if a.element == self.anchor_atom]

    def align_structures(self) -> None:
        """Align all structures to the first (reference) via anchor atoms."""
        if not self.structures:
            raise ValueError("No structures loaded.")
        ref     = self.structures[0]
        aligned = [ref]
        for tgt in self.structures[1:]:
            aligned.append(self._align_pair(ref, tgt))
        self.structures = aligned

    def _align_pair(self, ref: List[AtomPlus], tgt: List[AtomPlus]) -> List[AtomPlus]:
        ref_pts = np.array([[a.x,a.y,a.z] for a in self.get_anchor_atoms(ref)])
        tgt_pts = np.array([[a.x,a.y,a.z] for a in self.get_anchor_atoms(tgt)])
        n       = min(len(ref_pts), len(tgt_pts))
        if n == 0:
            raise ValueError("No anchor atoms for alignment.")
        ref_pts, tgt_pts = ref_pts[:n], tgt_pts[:n]

        # pick translation+rotation by n=1,2 or Kabsch
        if n == 1:
            R = np.eye(3)
            t = ref_pts[0] - tgt_pts[0]
        elif n == 2:
            mid_r, mid_t = ref_pts.mean(0), tgt_pts.mean(0)
            v_r = ref_pts[1] - ref_pts[0]
            v_t = tgt_pts[1] - tgt_pts[0]
            axis = np.cross(v_t/np.linalg.norm(v_t), v_r/np.linalg.norm(v_r))
            if np.linalg.norm(axis) < 1e-6:
                R = np.eye(3)
            else:
                ang = np.arccos(np.clip(
                    np.dot(v_t, v_r) / (np.linalg.norm(v_t)*np.linalg.norm(v_r)), -1, 1))
                R = rotation_matrix(axis, ang)
            t = mid_r - R @ mid_t
        else:
            cr, ct = ref_pts.mean(0), tgt_pts.mean(0)
            H       = (tgt_pts - ct).T @ (ref_pts - cr)
            U, S, Vt= np.linalg.svd(H)
            R       = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1,:] *= -1
                R         = Vt.T @ U.T
            t = cr - R @ ct

        aligned = []
        for a in tgt:
            v  = np.array([a.x,a.y,a.z])
            nv = R @ v + t
            aligned.append(AtomPlus(a.atom_id, a.atom_name, a.residue_name,
                                     a.residue_number, nv[0], nv[1], nv[2], a.element))
        return aligned

    def compute_histogram_descriptor(
        self, bins: int = 50, rmin: float = None, rmax: float = None
    ) -> np.ndarray:
        desc = []
        for struct in self.structures:
            coords = np.array([[a.x,a.y,a.z] for a in struct])
            dmat   = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)
            iu     = np.triu_indices_from(dmat, k=1)
            vals   = dmat[iu]
            lo     = vals.min() if rmin is None else rmin
            hi     = vals.max() if rmax is None else rmax
            hist, _= np.histogram(vals, bins=bins, range=(lo, hi), density=True)
            desc.append(hist)
        return np.vstack(desc)

    def compute_rmsd_descriptor(self) -> np.ndarray:
        ref   = np.array([[a.x,a.y,a.z] for a in self.structures[0]])
        rmsds = []
        for struct in self.structures:
            tgt  = np.array([[a.x,a.y,a.z] for a in struct])
            diff = ref - tgt
            rmsds.append(np.sqrt((diff**2).sum(axis=1).mean()))
        return np.array(rmsds)

    def compute_inertia_descriptor(self) -> np.ndarray:
        inert = []
        for struct in self.structures:
            coords = np.array([[a.x,a.y,a.z] for a in struct])
            cm     = coords.mean(0)
            rel    = coords - cm
            I      = np.zeros((3,3))
            for x,y,z in rel:
                I[0,0] += y*y + z*z
                I[1,1] += x*x + z*z
                I[2,2] += x*x + y*y
                I[0,1] -= x*y
                I[0,2] -= x*z
                I[1,2] -= y*z
            I[1,0], I[2,0], I[2,1] = I[0,1], I[0,2], I[1,2]
            vals,_ = np.linalg.eigh(I)
            inert.append(np.sort(vals))
        return np.vstack(inert)

    def compute_element_centers(self) -> List[Dict[str, np.ndarray]]:
        """
        For each aligned structure, compute the geometric center of each element type,
        **relative** to the anchor‐atom center.  Also builds:
          - self.element_types: sorted list of symbols
          - self.center_dists:  array (n_structures × n_element_types) of Euclidean norms
        """
        centers = []
        for struct in self.structures:
            anchors = [a for a in struct if a.element == self.anchor_atom]
            if not anchors:
                raise ValueError("No anchor atoms found for computing centers.")
            anc_coords    = np.array([[a.x,a.y,a.z] for a in anchors])
            anchor_center = anc_coords.mean(axis=0)

            elem_centers: Dict[str, np.ndarray] = {}
            for el in set(a.element for a in struct):
                coords      = np.array([[a.x,a.y,a.z] for a in struct if a.element == el])
                geom_center = coords.mean(axis=0)
                elem_centers[el] = geom_center - anchor_center

            centers.append(elem_centers)

        # store element_centers
        self.element_centers = centers

        # record element types in a consistent order
        self.element_types = sorted(centers[0].keys())

        # build center_dists array
        nstruc = len(centers)
        nelem  = len(self.element_types)
        cdists = np.zeros((nstruc, nelem))
        for i, ctr in enumerate(centers):
            for j, el in enumerate(self.element_types):
                cdists[i, j] = np.linalg.norm(ctr[el])
        self.center_dists = cdists

        return centers

    def match_atoms_to_reference(
        self, ref: List[AtomPlus], tgt: List[AtomPlus]
    ) -> Dict[int,int]:
        rd, td = {}, {}
        for i,a in enumerate(ref): rd.setdefault(a.element, []).append((i, np.array([a.x,a.y,a.z])))
        for i,a in enumerate(tgt): td.setdefault(a.element, []).append((i, np.array([a.x,a.y,a.z])))
        m = {}
        for el, rat in rd.items():
            tat = td.get(el)
            if tat is None or len(tat) != len(rat):
                raise ValueError(f"Mismatch element {el}")
            n    = len(rat)
            cost = np.zeros((n,n))
            for i, (_, rp) in enumerate(rat):
                for j, (_, tp) in enumerate(tat):
                    cost[i,j] = np.linalg.norm(rp-tp)
            rids, cids = linear_sum_assignment(cost)
            for i,j in zip(rids, cids):
                m[rat[i][0]] = tat[j][0]
        return m

    def match_all_atoms(self) -> List[Dict[int,int]]:
        ref = self.structures[0]
        return [self.match_atoms_to_reference(ref, tgt) for tgt in self.structures[1:]]

    def compute_coordination_numbers(self) -> List[Dict[Tuple[str,str], List[int]]]:
        """
        For each structure, compute coordination numbers for each (absorber, neighbor) pair
        based on self.coord_cutoffs, storing raw counts and sorted sets.
        """
        if not self.coord_cutoffs:
            return []

        cn_list = []
        sets_list = []
        for struct in self.structures:
            absorbers = [a for a in struct if a.element == self.anchor_atom]
            cn_dict = {}
            set_dict = {}
            for pair, cutoff in self.coord_cutoffs.items():
                abs_el, nbr_el = pair
                counts = []
                for a in absorbers:
                    # collect neighbor atoms, excluding self if same element
                    nbrs = [
                        b for b in struct
                        if b.element == nbr_el and not (abs_el == nbr_el and b.atom_id == a.atom_id)
                    ]
                    dists = [
                        np.linalg.norm(np.array([a.x,a.y,a.z]) - np.array([b.x,b.y,b.z]))
                        for b in nbrs
                    ]
                    counts.append(int(np.sum(np.array(dists) <= cutoff)))
                cn_dict[pair] = counts
                # unordered set: sort counts descending for comparison
                set_dict[pair] = tuple(sorted(counts, reverse=True))
            cn_list.append(cn_dict)
            sets_list.append(set_dict)

        self.coordination_numbers = cn_list
        self.coordination_sets = sets_list
        return cn_list
    
    def plot_coordination_histograms(
        self,
        figsize: Tuple[float,float] = None
    ) -> None:
        """
        For each atom-atom pair, plot a bar chart of coordination-number sets frequency.
        """
        if not self.coordination_sets:
            raise RuntimeError("Compute coordination numbers first.")
        pairs = list(self.coord_cutoffs.keys())
        n_pairs = len(pairs)
        if figsize is None:
            figsize = (6, 3 * n_pairs)
        fig, axes = plt.subplots(n_pairs, 1, figsize=figsize)
        if n_pairs == 1:
            axes = [axes]
        for ax, pair in zip(axes, pairs):
            # tally set occurrences
            all_sets = [s[pair] for s in self.coordination_sets]
            labels, counts = np.unique(all_sets, return_counts=True, axis=0)
            # prepare labels
            lbls = ["(" + ",".join(map(str, lbl)) + ")" for lbl in labels]
            ax.bar(lbls, counts)
            ax.set_ylabel("Count")
            ax.set_title(f"Coordination sets for {pair[0]}–{pair[1]}")
        plt.tight_layout()
        plt.show()
        
    def plot_element_center_histograms(
        self,
        bins: int = 20,
        hist_range: Tuple[float, float] = None,
        figsize: Tuple[float, float] = None
    ) -> None:
        """
        For each non-anchor element type, plot a histogram of the distance between that
        element’s geometric center and the anchor‐atom geometric center across all structures.
        """
        if not self.element_centers:
            raise RuntimeError("You must run() first so that element_centers are available.")

        anchor = self.anchor_atom
        # sanity check anchor–anchor
        anchor_dists = self.center_dists[:, self.element_types.index(anchor)]
        print(f"[sanity] anchor '{anchor}' centroid dists: min={anchor_dists.min():.3e}, "
              f"max={anchor_dists.max():.3e}")
        if not np.allclose(anchor_dists, 0, atol=1e-6):
            print("  ⚠️ Some anchor–anchor centroid distances are nonzero!")

        # plot all except anchor
        elems_to_plot = [e for e in self.element_types if e != anchor]
        n_elem = len(elems_to_plot)
        if n_elem == 0:
            print("No non-anchor element types found.")
            return

        if figsize is None:
            figsize = (6, 2.5 * n_elem)

        plt.close('all')
        fig, axes = plt.subplots(n_elem, 1, figsize=figsize, sharex=True)
        if n_elem == 1:
            axes = [axes]

        for ax, el in zip(axes, elems_to_plot):
            idx = self.element_types.index(el)
            ax.hist(self.center_dists[:, idx], bins=bins, range=hist_range, alpha=0.75)
            ax.set_ylabel("Count")
            ax.set_title(f"{el} → {anchor} center dists")

        axes[-1].set_xlabel("Distance (Å)")
        plt.tight_layout()
        plt.show()

    def plot_alignment(self) -> None:
        """
        Interactive 3D alignment viewer:
          ←/→ navigate targets, m=measure, r=reset, Esc=exit
        """
        if len(self.structures) < 2:
            print("Need at least two structures to visualize alignment.")
            return
        ref = self.structures[0]
        maps = self.match_all_atoms()
        total = len(maps)

        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        fig.text(0.5, 0.02,
                 "←/→ navigate   m=measure   r=reset   Esc=exit",
                 ha='center', va='bottom', fontsize=8)

        # plot reference
        ref_anc = np.array([[a.x,a.y,a.z] for a in ref if a.element==self.anchor_atom])
        ref_oth = np.array([[a.x,a.y,a.z] for a in ref if a.element!=self.anchor_atom])
        ref_anchor_scatter = ax.scatter(*ref_anc.T, c='blue', marker='^', s=80, label='Ref Anchor')
        ref_other_scatter  = None
        if ref_oth.size:
            ref_other_scatter = ax.scatter(*ref_oth.T, c='blue', marker='o', s=80, label='Ref Other')

        current = 0
        target_anchor_scatter = None
        target_other_scatter  = None
        lines = []
        measure_mode = False
        measure_pts = []
        meas_line = None
        meas_text = None
        highlights = []

        def clear_meas():
            nonlocal measure_pts, meas_line, meas_text, highlights
            measure_pts = []
            if meas_line: meas_line.remove()
            if meas_text: meas_text.remove()
            for hl in highlights: hl.remove()
            highlights.clear()
            fig.canvas.draw()

        def toggle_meas():
            nonlocal measure_mode
            measure_mode = not measure_mode
            if not measure_mode: clear_meas()
            print(f"Measurement mode {'ON' if measure_mode else 'OFF'}")

        def on_pick(event):
            nonlocal measure_pts, meas_line, meas_text, highlights
            if not measure_mode: return
            data = getattr(event.artist, '_data3d', None)
            if data is None: return
            coord = data[event.ind[0]]
            hl = ax.scatter(*coord, c='yellow', s=120, edgecolor='black')
            highlights.append(hl)
            measure_pts.append(coord)
            if len(measure_pts)==2:
                p1,p2 = measure_pts
                dist = np.linalg.norm(p1-p2)
                meas_line, = ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], 'k-', lw=2)
                mid = (p1+p2)/2
                meas_text = ax.text(*mid, f"{dist:.3f}", fontsize=9)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)

        def update(idx):
            nonlocal target_anchor_scatter, target_other_scatter, lines
            clear_meas()
            if target_anchor_scatter: target_anchor_scatter.remove()
            if target_other_scatter: target_other_scatter.remove()
            for ln in lines: ln.remove()
            lines.clear()

            mapping = maps[idx]
            tgt = self.structures[idx+1]
            ref_pts = []
            tgt_pts = []
            anc_ref, anc_tgt, oth_tgt = [], [], []
            for ri in sorted(mapping):
                ra = ref[ri]
                ta = tgt[mapping[ri]]
                ref_pts.append([ra.x,ra.y,ra.z])
                tgt_pts.append([ta.x,ta.y,ta.z])
                if ra.element==self.anchor_atom:
                    anc_ref.append([ra.x,ra.y,ra.z])
                    anc_tgt.append([ta.x,ta.y,ta.z])
                else:
                    oth_tgt.append([ta.x,ta.y,ta.z])

            ref_arr = np.array(ref_pts)
            tgt_arr = np.array(tgt_pts)
            offset = (np.mean(anc_ref,axis=0) - np.mean(anc_tgt,axis=0)) if anc_ref else np.zeros(3)
            tgt_arr += offset

            ta_arr = (np.array(anc_tgt)+offset) if anc_tgt else np.empty((0,3))
            to_arr = (np.array(oth_tgt)+offset) if oth_tgt else np.empty((0,3))
            target_anchor_scatter = ax.scatter(*ta_arr.T, c='red', marker='^', s=80,
                                               label='Tgt Anchor', picker=5)
            target_other_scatter  = ax.scatter(*to_arr.T,  c='red', marker='o', s=80,
                                               label='Tgt Other',  picker=5)
            target_anchor_scatter._data3d = ta_arr
            target_other_scatter._data3d  = to_arr

            for p_ref,p_tgt in zip(ref_arr, tgt_arr):
                for ln in ax.plot([p_ref[0],p_tgt[0]], [p_ref[1],p_tgt[1]], [p_ref[2],p_tgt[2]], 'k--', lw=0.5):
                    lines.append(ln)

            handles = [ref_anchor_scatter]
            labels  = ['Reference Anchor']
            if ref_other_scatter:
                handles.append(ref_other_scatter)
                labels.append('Reference Other')
            handles += [target_anchor_scatter, target_other_scatter]
            labels  += ['Target Anchor','Target Other']
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05,0.5))

            fname = self.filenames[idx+1]
            ax.set_title(f"Structure: {fname} ({idx+1}/{total})", fontsize=10)
            fig.canvas.draw()

        def on_key(evt):
            nonlocal current
            if evt.key=='right':
                current=(current+1)%total; update(current)
            elif evt.key=='left':
                current=(current-1)%total; update(current)
            elif evt.key=='m':
                toggle_meas()
            elif evt.key=='r':
                clear_meas()
            elif evt.key=='escape':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        update(0)
        plt.show(block=True)

    def plot_rmsd(self) -> None:
        """Scatter plot of per-structure RMSD."""
        if self.rmsd is None:
            raise RuntimeError("RMSD not computed; run() first.")
        plt.figure()
        idxs = np.arange(len(self.rmsd))
        plt.scatter(idxs, self.rmsd, marker='o')
        plt.xlabel("Structure Index")
        plt.ylabel("RMSD")
        plt.title("RMSD per Structure")
        plt.grid(True)
        plt.show()

    def plot_inertia(self) -> None:
        """Scatter plot of principal moments of inertia."""
        if self.inertia is None:
            raise RuntimeError("Inertia not computed; run() first.")
        fig, ax = plt.subplots()
        idxs = np.arange(len(self.inertia))
        labels = ["I₁","I₂","I₃"]
        for i,l in enumerate(labels):
            ax.scatter(idxs, self.inertia[:,i], marker='o', label=l)
        ax.set_xlabel("Structure Index")
        ax.set_ylabel("Principal Moment")
        ax.set_title("Moments of Inertia")
        ax.legend()
        ax.grid(True)
        plt.show()

    def run(
        self,
        methods: Union[List[str], str]      = ['histogram','rmsd','inertia'],
        histogram_bins: int                 = 50,
        histogram_range: Tuple[float,float] = (None,None),
        verbose: bool                       = False
    ) -> Dict[str, np.ndarray]:
        """
        Full workflow:
          1. Load
          2. Align
          3. Compute element centers
          4. Compute coordination numbers (if cutoffs provided)
          5. Compute descriptors
        """
        plt.close('all')
        if verbose: print("[1] Reading structures…")
        self.read_structures()
        if verbose: print("[2] Aligning…")
        self.align_structures()
        if verbose: print("[3] Computing element centers…")
        self.compute_element_centers()

        if self.coord_cutoffs:
            if verbose: print("[4] Computing coordination numbers…")
            self.compute_coordination_numbers()

        if verbose: print("[5] Computing descriptors…")
        results: Dict[str, np.ndarray] = {}
        if isinstance(methods, str): methods = [methods]
        if 'histogram' in methods:
            lo, hi = histogram_range
            self.histogram = self.compute_histogram_descriptor(
                bins=histogram_bins, rmin=lo, rmax=hi)
            results['histogram'] = self.histogram
        if 'rmsd' in methods:
            self.rmsd = self.compute_rmsd_descriptor()
            results['rmsd'] = self.rmsd
        if 'inertia' in methods:
            self.inertia = self.compute_inertia_descriptor()
            results['inertia'] = self.inertia

        if verbose:
            print("[6] Done. Descriptors, centers, and coordination numbers available.")
        return results