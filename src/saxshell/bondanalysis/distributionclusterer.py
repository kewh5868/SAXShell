import os
import re
import shutil
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Optional: k-medoids
try:
    from sklearn_extra.cluster import KMedoids

    HAVE_KMEDOIDS = True
except ImportError:
    HAVE_KMEDOIDS = False

from motif.structuredescriptor import StructureDescriptor


def remove_report(root_folder: str) -> None:
    """Delete all report_<datetime>/ subdirectories (and their contents)
    from each direct subfolder of `root_folder`.

    A report directory is identified by the pattern: report_YYYYMMDD_HHMMSS

    Parameters
    ----------
    root_folder : str
        Path whose immediate subfolders will be scanned for report_*
        directories to delete.
    """
    # Regex for report_<YYYYMMDD>_<HHMMSS>
    report_pattern = re.compile(r"^report_\d{8}_\d{6}$")

    # Iterate over each item in the root folder
    for entry in os.listdir(root_folder):
        subdir = os.path.join(root_folder, entry)
        if not os.path.isdir(subdir):
            continue

        # Look for report_<datetime> folders inside this subdir
        for name in os.listdir(subdir):
            if report_pattern.match(name):
                report_path = os.path.join(subdir, name)
                try:
                    shutil.rmtree(report_path)
                    print(f"Deleted report folder: {report_path}")
                except Exception as e:
                    print(f"Failed to delete {report_path}: {e}")


class DistributionClusterer:
    """Cluster a set of structures (PDB/XYZ) based on structural
    descriptors.

    Available comparators:
      • 'rmsd'         – RMSD to the reference structure
      • 'inertia'      – Three principal moments of inertia
      • 'centroids'    – Distances from each element’s centre to anchor-atom centre
      • 'coordination' – Coordination numbers for specified absorber–partner pairs
    """

    comparator_descriptions = {
        "rmsd": "Root-mean-square deviation to reference (StructureDescriptor)",
        "inertia": "Three principal moments of inertia (StructureDescriptor)",
        "centroids": "Distances from each element’s centre to anchor-atom centre",
        "coordination": "Coordination numbers for absorber–partner pairs (StructureDescriptor)",
    }

    def __init__(
        self,
        structure_folder: str,
        comparators: List[str] = None,
        n_clusters: int = None,
        cluster_method: str = "kmeans",
        pca_components: Optional[int] = None,
        silhouette_range: Tuple[int, int] = (2, 10),
        random_state: int = 42,
        anchor_atom: str = "Pb",
        comparator_weights: Dict[str, float] = None,
        coord_cutoffs: Dict[Tuple[str, str], float] = None,
    ):
        self.structure_folder = structure_folder
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self.pca_components = pca_components
        self.silhouette_range = silhouette_range
        self.random_state = random_state
        self.anchor_atom = anchor_atom
        self.coord_cutoffs = coord_cutoffs or {}

        # comparators
        all_keys = list(self.comparator_descriptions.keys())
        self.comparators = comparators or all_keys
        default_w = {k: 1.0 for k in all_keys}
        if comparator_weights:
            for k, v in comparator_weights.items():
                if k not in default_w:
                    raise KeyError(f"Unknown comparator '{k}'")
                default_w[k] = float(v)
        self.comparator_weights = default_w

        # placeholders
        self.filenames: List[str] = []
        self.struct_rmsd = None
        self.struct_inertia = None
        self.struct_centroids = None
        self.centroid_types: List[str] = []
        self.struct_coordination = None
        self.coordination_pairs: List[Tuple[str, str]] = []
        self.features = None
        self.labels = None
        self.centroids = None
        self.medoid_indices = None
        self.rep_indices = None
        self.tested_ks: List[int] = []
        self.inertias: List[float] = []
        self.silhouette_scores = []

    def describe_comparators(self):
        """Print available comparators and which are active."""
        print("Available comparators:")
        for key, desc in self.comparator_descriptions.items():
            mark = "✔" if key in self.comparators else "✘"
            print(f"  {mark} {key:12s} – {desc}")

    def _load_structural_descriptors(self):
        """Compute descriptors using StructureDescriptor."""
        files = sorted(
            f
            for f in os.listdir(self.structure_folder)
            if f.endswith((".xyz", ".pdb"))
        )
        if not files:
            raise ValueError(
                f"No structure files found in {self.structure_folder}"
            )
        self.filenames = files

        sd = StructureDescriptor(
            self.structure_folder,
            anchor_atom=self.anchor_atom,
            coord_cutoffs=self.coord_cutoffs,
        )
        methods = [m for m in ("rmsd", "inertia") if m in self.comparators]
        sd.run(methods=methods, verbose=False)

        if "rmsd" in self.comparators:
            self.struct_rmsd = sd.rmsd
        if "inertia" in self.comparators:
            self.struct_inertia = sd.inertia
        if "centroids" in self.comparators:
            ecs = sd.compute_element_centers()
            types = sorted(e for e in ecs[0].keys() if e != self.anchor_atom)
            self.centroid_types = types
            self.struct_centroids = np.vstack(
                [[np.linalg.norm(c[e]) for e in types] for c in ecs]
            )
        if "coordination" in self.comparators:
            cns_list = sd.compute_coordination_numbers()
            self.struct_coordination = cns_list
            self.coordination_pairs = list(cns_list[0].keys())

    def _assemble_features(self):
        """Create feature matrix from descriptors."""
        rows = []
        for i in range(len(self.filenames)):
            blocks = []
            for comp in self.comparators:
                w = self.comparator_weights[comp]
                if comp == "rmsd":
                    blocks.append(np.array([self.struct_rmsd[i] * w]))
                elif comp == "inertia":
                    blocks.append(self.struct_inertia[i] * w)
                elif comp == "centroids":
                    blocks.append(self.struct_centroids[i] * w)
                elif comp == "coordination":
                    vals = []
                    for pair in self.coordination_pairs:
                        arr = np.array(
                            self.struct_coordination[i][pair], dtype=int
                        )
                        vals.extend(np.sort(arr))
                    blocks.append(np.array(vals) * w)
            rows.append(np.hstack(blocks))
        self.features = np.vstack(rows)

    def preprocess(self):
        """Standardize & optionally PCA reduce."""
        if self.features is None:
            raise ValueError(
                "Features have not been assembled. Call _assemble_features() first."
            )
        scaler = StandardScaler()
        X = scaler.fit_transform(self.features)
        if self.pca_components:
            max_dim = min(X.shape)
            n_comp = min(self.pca_components, max_dim)
            if self.pca_components > max_dim:
                warnings.warn(
                    f"PCA comps ({self.pca_components}) > dims ({max_dim}); using {max_dim}"
                )
            if n_comp > 0:
                X = PCA(
                    n_components=n_comp, random_state=self.random_state
                ).fit_transform(X)
        self.features = X

    def determine_k(self):
        """Auto-select number of clusters or use preset."""
        if self.n_clusters is not None:
            return self.n_clusters
        self.silhouette_scores = []
        best_k, best_score = None, -1
        self.tested_ks, self.inertias = [], []
        kmin, kmax = self.silhouette_range
        for k in range(kmin, kmax + 1):
            if self.cluster_method == "kmedoids" and HAVE_KMEDOIDS:
                m = KMedoids(n_clusters=k, random_state=self.random_state)
            else:
                m = KMeans(n_clusters=k, random_state=self.random_state)
            labs = m.fit_predict(self.features)
            score = silhouette_score(self.features, labs)
            self.tested_ks.append(k)
            self.inertias.append(m.inertia_)
            if score > best_score:
                best_score, best_k = score, k
            self.silhouette_scores.append(score)
        self.n_clusters = best_k
        return best_k

    def cluster(self):
        """Perform clustering."""
        if self.n_clusters is None:
            raise ValueError("n_clusters not set. Call determine_k() first.")
        k = self.determine_k()
        if self.cluster_method == "kmedoids":
            if not HAVE_KMEDOIDS:
                raise ImportError("sklearn_extra required for kmedoids")
            med = KMedoids(n_clusters=k, random_state=self.random_state)
            self.labels = med.fit_predict(self.features)
            self.medoid_indices = med.medoid_indices_
        else:
            km = KMeans(n_clusters=k, random_state=self.random_state)
            self.labels = km.fit_predict(self.features)
            self.centroids = km.cluster_centers_
        return self.labels

    # def select_representatives(self):
    #     """Choose representative structure per cluster."""
    #     reps = []
    #     if self.cluster_method == 'kmedoids' and self.medoid_indices is not None:
    #         reps = list(self.medoid_indices)
    #     else:
    #         for i in range(self.n_clusters):
    #             idxs = np.where(self.labels == i)[0]
    #             center = self.centroids[i]
    #             dists = np.linalg.norm(self.features[idxs] - center, axis=1)
    #             reps.append(idxs[np.argmin(dists)])
    #     self.rep_indices = reps
    #     return reps

    def write_report(self):
        """Write summary report."""
        report_path = os.path.join(self.report_dir, "cluster_report.txt")
        sil = (
            silhouette_score(self.features, self.labels)
            if self.n_clusters > 1
            else None
        )
        weights = [
            int(np.sum(self.labels == i)) for i in range(self.n_clusters)
        ]
        with open(report_path, "w") as f:
            f.write(f"Run Report\nDate/Time: {datetime.now().isoformat()}\n\n")
            f.write("Comparators used:\n")
            for key in self.comparators:
                f.write(
                    f"  - {key:12s}: {self.comparator_descriptions[key]}\n"
                )

            # Comparator weights
            f.write("\nComparator weights:\n")
            for key, w in self.comparator_weights.items():
                if key in self.comparators:
                    f.write(f"  - {key:12s}: {w}\n")
            # Coordination cutoffs
            f.write("\nCoordination cutoffs (if any):\n")
            if self.coord_cutoffs:
                for pair, cutoff in self.coord_cutoffs.items():
                    f.write(f"  - {pair}: {cutoff}\n")
            else:
                f.write("  None\n")

            f.write(f"\nClustering method : {self.cluster_method}\n")
            f.write(f"Number of clusters: {self.n_clusters}\n")
            if sil is not None:
                f.write(f"Silhouette score  : {sil:.4f}\n")
            f.write("\nCluster sizes (# members):\n")
            for i, w in enumerate(weights):
                f.write(f"  Cluster {i:2d}: {w}\n")
            # f.write("\nRepresentatives:\n")
            # for i, rep in enumerate(self.rep_indices):
            #     f.write(f"  Cluster {i:2d}: {self.filenames[rep]}\n")
        return report_path

    def write_cluster_membership(self):
        """Write membership details."""
        membership = os.path.join(self.report_dir, "cluster_membership.txt")
        with open(membership, "w") as f:
            for i in range(self.n_clusters):
                f.write(f"Cluster {i}\n")
                # f.write(f"Representative: {self.filenames[self.rep_indices[i]]}\n")
                for j in np.where(self.labels == i)[0]:
                    # if j != self.rep_indices[i]:
                    f.write(f"  Member: {self.filenames[j]}\n")
                f.write("\n")
        return membership

    # def copy_representatives(self):
    #     """Copy representative files to report directory."""
    #     copied = []
    #     for rep in self.rep_indices:
    #         fname = self.filenames[rep]
    #         src = os.path.join(self.structure_folder, fname)
    #         dst = os.path.join(self.report_dir, fname)
    #         if os.path.exists(src):
    #             shutil.copy(src, dst)
    #             copied.append(dst)
    #         else:
    #             warnings.warn(f"Representative file not found: {src}")
    #     return copied

    def plot_elbow(self):
        """Plot elbow curve."""
        plt.figure()
        plt.plot(self.tested_ks, self.inertias, "o-", lw=2)
        plt.axvline(self.n_clusters, ls="--", color="k")
        plt.xlabel("k")
        plt.ylabel("Inertia (WSS)")
        plt.title("Elbow Method")
        plt.show()

    def plot_silhouette(self):
        """Plot silhouette score vs k for selected comparator set."""
        if not self.tested_ks or not self.silhouette_scores:
            raise ValueError("No silhouette data. Run determine_k() first.")
        plt.figure()
        plt.plot(self.tested_ks, self.silhouette_scores, "o-", lw=2)
        plt.axvline(self.n_clusters, ls="--", color="k")
        plt.xlabel("k")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis")
        plt.show()

    def run(self, verbose: bool = False):
        """Full clustering pipeline."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(self.structure_folder, f"report_{ts}")
        os.makedirs(self.report_dir, exist_ok=True)
        if verbose:
            print(f"Report dir: {self.report_dir}")

        if verbose:
            print("Loading structural descriptors...")
        self._load_structural_descriptors()
        if verbose:
            print("Assembling features...")
        self._assemble_features()
        self.describe_comparators()
        if verbose:
            print("Preprocessing...")
        self.preprocess()
        if verbose:
            print("Determining k...")
        self.determine_k()
        if verbose:
            print(f"Using k = {self.n_clusters}")
        if verbose:
            print("Clustering...")
        self.cluster()
        # if verbose: print("Selecting representatives...")
        # self.select_representatives()
        if verbose:
            print("Writing reports...")
        report_path = self.write_report()
        membership_path = self.write_cluster_membership()
        # if verbose: print("Copying representatives...")
        # copied = self.copy_representatives()
        if verbose:
            print(f"Reports written to: {self.report_dir}")
            print(f" - Summary: {report_path}")
            print(f" - Membership: {membership_path}")
            # if copied:
            #     print(f"Representative structures copied: {len(copied)} files")
        if verbose:
            print("Plotting elbow...")
        # try:
        #     self.plot_elbow()
        # except Exception as e:
        #     warnings.warn(f"Elbow plot failed: {e}")
        if verbose:
            print("Done.")
