import csv
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class DisplacementAnalysis:
    """Analyzes cluster folders for bond/angle (distribution/angle)
    files, labels each as 'Bond Pair' or 'Angle Triplet', and loads the
    relevant data."""

    def __init__(self, analyzed_clusters_dir):
        self.ana_dir = os.path.abspath(analyzed_clusters_dir)
        self.structure_folders = self._find_structure_folders()
        # Map: structure name → { 'Bond Pair': {pair: filepath}, 'Angle Triplet': {triplet: filepath} }
        self.structure_distribution_map = {}
        # Unique and shared labels (with type)
        self.unique_bond_pairs = set()
        self.shared_bond_pairs = set()
        self.unique_angle_triplets = set()
        self.shared_angle_triplets = set()
        # Loaded data: (structure, label) -> list of values (third column)
        self.data = {}
        self._scan_all_structures_and_load_data()

    def _find_structure_folders(self):
        return [
            os.path.join(self.ana_dir, entry)
            for entry in os.listdir(self.ana_dir)
            if os.path.isdir(os.path.join(self.ana_dir, entry))
            and not entry.startswith("representative_")
        ]

    def _scan_all_structures_and_load_data(self):
        all_bond_pairs = []
        all_angle_triplets = []
        self.structure_distribution_map = {}
        self.data = {}

        for struct_path in self.structure_folders:
            struct_name = os.path.basename(struct_path)
            outdist_path = os.path.join(struct_path, "output_distributions")
            bond_pairs = {}
            angle_triplets = {}
            if not os.path.isdir(outdist_path):
                print(f"[WARN] {struct_name} missing output_distributions/")
                continue

            for fname in os.listdir(outdist_path):
                fpath = os.path.join(outdist_path, fname)
                if not os.path.isfile(fpath):
                    continue
                # Bond Pairs (_distribution.csv)
                if fname.endswith("_distribution.csv"):
                    m = re.match(
                        r"^([A-Za-z0-9]+_[A-Za-z0-9]+)_distribution\.csv$",
                        fname,
                    )
                    if m:
                        pair = m.group(1)
                        label = (pair, "Bond Pair")
                        bond_pairs[pair] = fpath
                        # Load third column values
                        self.data[(struct_name, label)] = (
                            self._load_third_column_from_csv(fpath)
                        )
                # Angle Triplets (_angles.csv)
                elif fname.endswith("_angles.csv"):
                    m = re.match(
                        r"^([A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+)_angles\.csv$",
                        fname,
                    )
                    if m:
                        triplet = m.group(1)
                        label = (triplet, "Angle Triplet")
                        angle_triplets[triplet] = fpath
                        self.data[(struct_name, label)] = (
                            self._load_third_column_from_csv(fpath)
                        )
            self.structure_distribution_map[struct_name] = {
                "Bond Pair": bond_pairs,
                "Angle Triplet": angle_triplets,
            }
            if bond_pairs:
                all_bond_pairs.append(set(bond_pairs.keys()))
            if angle_triplets:
                all_angle_triplets.append(set(angle_triplets.keys()))

        # Unique and shared bond pairs / angle triplets
        self.unique_bond_pairs = (
            set().union(*all_bond_pairs) if all_bond_pairs else set()
        )
        self.unique_angle_triplets = (
            set().union(*all_angle_triplets) if all_angle_triplets else set()
        )
        self.shared_bond_pairs = (
            set.intersection(*all_bond_pairs) if all_bond_pairs else set()
        )
        self.shared_angle_triplets = (
            set.intersection(*all_angle_triplets)
            if all_angle_triplets
            else set()
        )

    def _load_third_column_from_csv(self, filepath):
        """Extracts the third column (as floats) from a csv, skipping
        header row."""
        col = []
        try:
            with open(filepath, newline="") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 3:
                        try:
                            val = float(row[2])
                            col.append(val)
                        except Exception:
                            continue
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
        return col

    def merge_shared_distributions_and_angles(self):
        """For each shared bond pair or angle triplet, create a merged
        CSV containing all structure values, output to the
        representative_structures folder.

        Also keep all merged data in memory as self.merged_data.
        """
        # Find representative_structures or representative_clusters folder
        rep_folder = None
        for cand in ["representative_structures", "representative_clusters"]:
            rep_path = os.path.join(self.ana_dir, cand)
            if os.path.isdir(rep_path):
                rep_folder = rep_path
                break
        if not rep_folder:
            raise FileNotFoundError(
                "No representative_structures or representative_clusters folder found."
            )

        self.merged_data = {}

        # Merge bond pairs
        for pair in self.shared_bond_pairs:
            merged = []
            for struct in self.structure_distribution_map:
                key = (struct, (pair, "Bond Pair"))
                vals = self.data.get(key, [])
                for v in vals:
                    merged.append((struct, v))
            outname = f"{pair}_merged.csv"
            outpath = os.path.join(rep_folder, outname)
            with open(outpath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Structure", "Value"])
                writer.writerows(merged)
            print(f"[INFO] Wrote merged file: {outpath}")
            self.merged_data[outname] = merged  # Store in memory

        # Merge angle triplets
        for triplet in self.shared_angle_triplets:
            merged = []
            for struct in self.structure_distribution_map:
                key = (struct, (triplet, "Angle Triplet"))
                vals = self.data.get(key, [])
                for v in vals:
                    merged.append((struct, v))
            outname = f"{triplet}_merged.csv"
            outpath = os.path.join(rep_folder, outname)
            with open(outpath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Structure", "Value"])
                writer.writerows(merged)
            print(f"[INFO] Wrote merged file: {outpath}")
            self.merged_data[outname] = merged  # Store in memory

    def print_summary(self):
        print("Structures scanned:")
        for struct, v in self.structure_distribution_map.items():
            print(f"  {struct}:")
            print(f"    Bond Pairs: {sorted(v['Bond Pair'].keys())}")
            print(f"    Angle Triplets: {sorted(v['Angle Triplet'].keys())}")
        print(
            f"\nUnique Bond Pairs ({len(self.unique_bond_pairs)}): {sorted(self.unique_bond_pairs)}"
        )
        print(
            f"Shared Bond Pairs ({len(self.shared_bond_pairs)}): {sorted(self.shared_bond_pairs)}"
        )
        print(
            f"\nUnique Angle Triplets ({len(self.unique_angle_triplets)}): {sorted(self.unique_angle_triplets)}"
        )
        print(
            f"Shared Angle Triplets ({len(self.shared_angle_triplets)}): {sorted(self.shared_angle_triplets)}"
        )
        print("\nSample data (first 3 values for one structure):")
        for key in self.data:
            struct, (label, typ) = key
            if self.data[key]:
                print(f"  [{struct} | {label} | {typ}]: {self.data[key][:3]}")
                break

    def get_max_bond_length_from_csv(
        self, csv_filepath: str, column_index: int = 2
    ) -> float:
        """Reads a merged CSV of bond lengths or angles and returns the
        maximum value in the specified column."""
        import csv

        values = []
        with open(csv_filepath, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if column_index >= len(headers):
                raise IndexError(
                    f"Column index {column_index} out of range for headers: {headers}"
                )
            key = headers[column_index]
            for row in reader:
                try:
                    values.append(float(row[key]))
                except ValueError:
                    continue
        if not values:
            raise ValueError(
                f"No numeric values found in column '{key}' of {csv_filepath}"
            )
        return max(values)

    def extract_bond_lengths_from_xyz_folder(
        self,
        xyz_folder: str,
        element_pair: tuple,
        cutoff: float = None,
        merged_csv_filepath: str = None,
        column_index: int = 2,
    ) -> dict:
        """Extracts bond lengths between two element types from each XYZ
        file in a folder.

        If no cutoff is provided, it will be computed from the merged
        CSV. Returns a dict mapping filenames to numpy arrays of
        distances.
        """
        import os

        if cutoff is None:
            if merged_csv_filepath:
                cutoff = self.get_max_bond_length_from_csv(
                    merged_csv_filepath, column_index
                )
            else:
                raise ValueError(
                    "Either 'cutoff' or 'merged_csv_filepath' must be provided"
                )

        elem1, elem2 = element_pair
        bond_lengths = {}
        for fname in os.listdir(xyz_folder):
            if not fname.endswith(".xyz"):
                continue
            path = os.path.join(xyz_folder, fname)
            with open(path, "r") as f:
                lines = f.readlines()
            atom_lines = lines[2:]
            types, coords = [], []
            for line in atom_lines:
                parts = line.split()
                if len(parts) < 4:
                    continue
                types.append(parts[0])
                coords.append([float(p) for p in parts[1:4]])
            types = np.array(types)
            coords = np.array(coords)
            idx1 = np.where(types == elem1)[0]
            idx2 = np.where(types == elem2)[0]
            distances = []
            for i in idx1:
                for j in idx2:
                    if elem1 == elem2 and j <= i:
                        continue
                    d = np.linalg.norm(coords[i] - coords[j])
                    if d <= cutoff:
                        distances.append(d)
            bond_lengths[fname] = np.array(distances)
        return bond_lengths

    def compute_bond_lengths_for_pair(
        self,
        element_pair,
        xyz_folder=None,
        merged_csv_filename=None,
        column_index=2,
        cutoff=None,
    ):
        """Compute bond lengths for a selected bond pair from XYZ files
        in each structure folder.

        Returns a dict: structure_name -> filename -> numpy array of bond lengths.
        """
        import os

        # Guess default merged file name if not specified
        if cutoff is None and merged_csv_filename is None:
            merged_csv_filename = (
                f"{element_pair[0]}_{element_pair[1]}_merged.csv"
            )

        # Path to merged CSV in representative folder
        rep_folder = None
        for cand in ["representative_structures", "representative_clusters"]:
            rep_path = os.path.join(self.ana_dir, cand)
            if os.path.isdir(rep_path):
                rep_folder = rep_path
                break
        if rep_folder is None:
            raise FileNotFoundError(
                "No representative_structures or representative_clusters folder found."
            )

        merged_csv_path = (
            os.path.join(rep_folder, merged_csv_filename)
            if merged_csv_filename
            else None
        )

        # Compute cutoff from merged CSV if not specified
        if (
            cutoff is None
            and merged_csv_path
            and os.path.exists(merged_csv_path)
        ):
            cutoff = self.get_max_bond_length_from_csv(
                merged_csv_path, column_index
            )
        elif cutoff is None:
            raise ValueError("Could not find merged CSV to determine cutoff.")

        results = {}
        for struct_path in self.structure_folders:
            struct_name = os.path.basename(struct_path)
            folder = xyz_folder or struct_path
            result = self.extract_bond_lengths_from_xyz_folder(
                xyz_folder=folder,
                element_pair=element_pair,
                cutoff=cutoff,
                merged_csv_filepath=merged_csv_path,
                column_index=column_index,
            )
            results[struct_name] = result
        return results

    def get_representative_motif_values(
        self, bond_or_angle, type_label, cutoff=None
    ):
        """Finds all motif representative XYZ files for the given bond
        pair or angle triplet.

        Returns: dict {motif_label: [values,...], ...} (skip motif if XYZ file not found)
        Warns if 1:1 mapping between output_distributions and representative_clusters is not found.
        """
        import math

        from scipy.spatial import cKDTree

        # Setup: folders
        rep_folder = None
        for cand in ["representative_structures", "representative_clusters"]:
            test = os.path.join(self.ana_dir, cand)
            if os.path.isdir(test):
                rep_folder = test
                break
        if not rep_folder:
            print(
                "[WARN] No representative_structures or representative_clusters folder found."
            )
            return {}

        motif_vals = {}
        # Loop all structures
        for struct in self.structure_distribution_map:
            outdist_path = os.path.join(
                self.ana_dir, struct, "output_distributions"
            )
            if not os.path.isdir(outdist_path):
                continue
            # Find all motif_X_*.pdb
            for pdbfile in os.listdir(outdist_path):
                # print(f"[DEBUG] Processing: {pdbfile}\n")
                m = re.match(r"(motif_(\d+))_(.+)\.pdb", pdbfile)
                if not m:
                    # print(f"[WARN] Skipping {pdbfile}: does not match motif pattern.\n")
                    continue

                motif_str, motif_num, basename = m.groups()
                motif_label = f"{struct}_{motif_str}"
                # Attempt to find corresponding XYZ file in rep_folder
                basename_no_ext = basename.rsplit(".", 1)[0]
                expected_xyz = f"{struct}_{motif_str}_{basename_no_ext}.xyz"
                xyz_path = os.path.join(rep_folder, expected_xyz)
                # print(f"[DEBUG] Looking for: {expected_xyz} in {rep_folder}")
                if not os.path.exists(xyz_path):
                    print(
                        f"[WARN] No matching XYZ for motif {motif_label} (expected: {expected_xyz})\n[NOTE] DisplacementAnalysis XYZ loader is not robust to deviations in XYZ file names - requires update."
                    )
                    continue
                # Load XYZ file
                types, coords = [], []
                with open(xyz_path, "r") as f:
                    lines = f.readlines()[2:]  # skip first 2
                for line in lines:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    types.append(parts[0])
                    coords.append([float(x) for x in parts[1:4]])
                types = np.array(types)
                coords = np.array(coords)
                values = []
                if type_label == "Bond Pair":
                    elem1, elem2 = bond_or_angle.split("_")
                    idx1 = np.where(types == elem1)[0]
                    idx2 = np.where(types == elem2)[0]
                    # Use cutoff if given, else just max distance in file
                    if cutoff is None:
                        dmax = 0
                        for i in idx1:
                            for j in idx2:
                                if elem1 == elem2 and j <= i:
                                    continue
                                d = np.linalg.norm(coords[i] - coords[j])
                                dmax = max(dmax, d)
                        use_cutoff = dmax * 1.01
                    else:
                        use_cutoff = cutoff
                    for i in idx1:
                        for j in idx2:
                            if elem1 == elem2 and j <= i:
                                continue
                            d = np.linalg.norm(coords[i] - coords[j])
                            if d <= use_cutoff:
                                values.append(d)
                elif type_label == "Angle Triplet":
                    ce, n1, n2 = bond_or_angle.split("_")
                    # No cutoff provided: use 3.0 Angstrom (arbitrary, or user can set)
                    cut1 = cut2 = cutoff if cutoff else 3.0
                    tree = cKDTree(coords)
                    for ci in (i for i, e in enumerate(types) if e == ce):
                        nbrs = tree.query_ball_point(
                            coords[ci], r=max(cut1, cut2)
                        )
                        for i in nbrs:
                            if types[i] != n1:
                                continue
                            d1 = np.linalg.norm(coords[i] - coords[ci])
                            if cut1 and d1 > cut1:
                                continue
                            for j in nbrs:
                                if j == i or types[j] != n2:
                                    continue
                                d2 = np.linalg.norm(coords[j] - coords[ci])
                                if cut2 and d2 > cut2:
                                    continue
                                v1 = coords[i] - coords[ci]
                                v2 = coords[j] - coords[ci]
                                cos = np.dot(v1, v2) / (
                                    np.linalg.norm(v1) * np.linalg.norm(v2)
                                )
                                ang = math.degrees(
                                    math.acos(np.clip(cos, -1.0, 1.0))
                                )
                                values.append(ang)
                if values:
                    motif_vals[motif_label] = values
        return motif_vals

    def load_md_prior_weights(self, filepath=None):
        """Loads the md_prior_weights.json file and maps (structure,
        motif) to its count, weight, and representative pdb.

        If filepath is not provided, defaults to md_prior_weights.json
        in the analysis directory. Result is stored in
        self.md_prior_weights_map as: {(structure, motif): {"count":
        int, "weight": float, "representative": str}}
        """
        if filepath is None:
            filepath = os.path.join(self.ana_dir, "md_prior_weights.json")
        if not os.path.exists(filepath):
            print(f"[WARN] md_prior_weights.json not found at {filepath}")
            self.md_prior_weights_map = {}
            return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            mapping = {}
            for structure, motif_dict in data.get("structures", {}).items():
                for motif, motif_info in motif_dict.items():
                    mapping[(structure, motif)] = {
                        "count": motif_info.get("count", None),
                        "weight": motif_info.get("weight", None),
                        "representative": motif_info.get(
                            "representative", None
                        ),
                    }
            self.md_prior_weights_map = mapping
            print(
                f"[INFO] Loaded md_prior_weights.json with {len(mapping)} (structure, motif) entries."
            )
        except Exception as e:
            print(f"[ERROR] Failed to load md_prior_weights.json: {e}")
            self.md_prior_weights_map = {}

    def plot_merged_distribution(
        self,
        bond_or_angle,
        type_label="Bond Pair",
        bins=100,
        show=True,
        ax=None,
        mode="histogram_positions",
    ):
        """Plot the merged distribution for a specified bond pair or
        angle triplet.

        mode options:
          - "histogram": Show only the merged histogram (no representative motif lines).
          - "histogram_positions": Histogram + thin black vertical line at the representative position for each motif.
          - "histogram_weighted": Histogram + black vertical line at each motif's representative position,
                                   with line thickness proportional to motif count (if available).
        """
        merged_filename = f"{bond_or_angle}_merged.csv"
        if (
            not hasattr(self, "merged_data")
            or merged_filename not in self.merged_data
        ):
            raise ValueError(
                f"Run merge_shared_distributions_and_angles first, or no merged data for {bond_or_angle}."
            )
        values = [v for s, v in self.merged_data[merged_filename]]
        total_hist_count = len(values)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        # Always show histogram
        ax.hist(values, bins=bins, alpha=0.5, color="tab:blue", label=None)

        # Get representative values per motif
        motif_vals = self.get_representative_motif_values(
            bond_or_angle, type_label
        )
        # Compute a single rep value per motif (e.g., mean)
        rep_positions = {}
        for motif_label, vals in motif_vals.items():
            if vals:
                rep_positions[motif_label] = np.mean(vals)

        if mode == "histogram":
            # nothing more
            pass
        elif mode == "histogram_positions":
            # thin black lines at each motif's representative position
            for pos in rep_positions.values():
                ax.axvline(pos, color="black", lw=1, linestyle="--", alpha=1.0)
        elif mode == "histogram_weighted":
            # Ensure weights loaded
            if not hasattr(self, "md_prior_weights_map"):
                self.load_md_prior_weights()
            motif_vals = self.get_representative_motif_values(
                bond_or_angle, type_label
            )
            # Compute one mean position per motif
            rep_positions = {
                ml: np.mean(vals) for ml, vals in motif_vals.items() if vals
            }
            # Gather motif counts
            counts = {}
            total_counts = 0
            for motif_label in rep_positions:
                if "_motif_" in motif_label:
                    struct, motif_num = motif_label.split("_motif_", 1)
                    motif_key = f"motif_{motif_num}"
                else:
                    struct, motif_key = motif_label, None
                cnt = (
                    self.md_prior_weights_map.get((struct, motif_key), {}).get(
                        "count", 0
                    )
                    or 0
                )
                counts[motif_label] = cnt
                total_counts += cnt
            if total_counts <= 0:
                print(
                    "[WARN] No motif counts available for weighted "
                    "mode; skipping motif lines."
                )
            else:
                # Get max bin height from histogram
                n, bins_, patches = ax.hist(
                    values, bins=bins, alpha=0
                )  # Just to get heights; alpha=0 hides them
                hist_max = np.max(n) if len(n) else 1
                # Calculate unscaled motif heights
                motif_heights = {
                    ml: (counts[ml] / total_counts) * total_hist_count
                    for ml in rep_positions
                }
                max_motif_height = (
                    max(motif_heights.values()) if motif_heights else 1
                )
                # Scaling factor to set max motif line to histogram max
                scale = (
                    hist_max / max_motif_height if max_motif_height > 0 else 1
                )
                # Draw solid black lines, scaled
                for motif_label, pos in rep_positions.items():
                    cnt = counts.get(motif_label, 0)
                    height = motif_heights[motif_label] * scale
                    ax.vlines(
                        pos,
                        0,
                        height,
                        color="black",
                        lw=1,
                        linestyle="-",
                        alpha=1.0,
                    )
        else:
            raise ValueError(
                "mode must be 'histogram', 'histogram_positions' or 'histogram_weighted'"
            )

        ax.set_xlabel(
            "Bond Length (Å)" if type_label == "Bond Pair" else "Angle (deg)"
        )
        ax.set_ylabel("Count")
        ax.set_title(f"Merged Distribution: {bond_or_angle} ({type_label})")
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def fit_histogram_with_weighted_gaussians_tk(
        self,
        bond_or_angle,
        type_label="Bond Pair",
        min_val=None,
        max_val=None,
        bins=100,
        components="lines",  # "lines", "gaussians", "none"
        show=False,  # If False, returns a matplotlib Figure for embedding in TK
    ):
        """Fits the merged histogram with a sum of Gaussians (see main
        method), but returns a matplotlib Figure for embedding into a TK
        window if show=False, or displays plot if show=True."""
        from matplotlib.figure import Figure

        # --- Collect motif values/weights as before ---
        motif_vals = self.get_representative_motif_values(
            bond_or_angle, type_label
        )
        if not hasattr(self, "md_prior_weights_map"):
            self.load_md_prior_weights()
        centers = []
        weights = []
        for motif_label, vals in motif_vals.items():
            if "_motif_" in motif_label:
                struct, motif_num = motif_label.split("_motif_", 1)
                motif_key = f"motif_{motif_num}"
            else:
                struct, motif_key = motif_label, None
            count = (
                self.md_prior_weights_map.get((struct, motif_key), {}).get(
                    "count", 0
                )
                or 0
            )
            for v in vals:
                if (min_val is None or v >= min_val) and (
                    max_val is None or v <= max_val
                ):
                    centers.append(v)
                    weights.append(count)
        centers = np.array(centers)
        weights = np.array(weights)
        if len(centers) == 0:
            print("No motif centers in range!")
            return None

        # --- Prepare histogram data ---
        merged_filename = f"{bond_or_angle}_merged.csv"
        values = np.array([v for _, v in self.merged_data[merged_filename]])
        if min_val is not None:
            values = values[values >= min_val]
        if max_val is not None:
            values = values[values <= max_val]
        counts, edges = np.histogram(
            values, bins=bins, range=(min_val, max_val)
        )
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Normalize motif weights to match total histogram area
        total_area = counts.sum() * (edges[1] - edges[0])
        motif_areas = weights / weights.sum() * total_area

        # --- Gaussian model setup ---
        def gauss(x, mu, sigma, area):
            return (
                area
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            )

        def sum_of_gaussians(x, sigma):
            return np.sum(
                [
                    gauss(x, mu, sigma, A)
                    for mu, A in zip(centers, motif_areas)
                ],
                axis=0,
            )

        # --- Fit ---
        sigma0 = np.std(values) / 2 if len(values) > 1 else 0.2
        try:
            popt, pcov = curve_fit(
                lambda x, sigma: sum_of_gaussians(x, sigma),
                bin_centers,
                counts,
                p0=[sigma0],
                bounds=([0.001], [10]),
            )
        except Exception as e:
            print("Fit failed:", e)
            return None

        sigma_fit = popt[0]
        fwhm_fit = 2.355 * sigma_fit
        sigma2_fit = sigma_fit**2

        # --- Plot, but return Figure for embedding if show==False ---
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.hist(
            values,
            bins=bins,
            range=(min_val, max_val),
            alpha=0.5,
            color="tab:blue",
            label="Merged histogram",
        )
        xfit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
        ax.plot(
            xfit,
            sum_of_gaussians(xfit, sigma_fit),
            "k-",
            label="Weighted Gaussian sum fit",
        )

        if components == "lines":
            for mu in centers:
                ax.axvline(mu, color="gray", lw=1, alpha=0.5)
        elif components == "gaussians":
            for i, (mu, area) in enumerate(zip(centers, motif_areas)):
                ax.plot(
                    xfit,
                    gauss(xfit, mu, sigma_fit, area),
                    "--",
                    lw=1.5,
                    alpha=0.7,
                    label=f"G{i+1}",
                )
        # (components == "none": do nothing extra)

        ax.set_xlabel(
            "Bond Length (Å)" if type_label == "Bond Pair" else "Angle (deg)"
        )
        ax.set_ylabel("Count")
        ax.set_title(
            f"Fit: {bond_or_angle}, FWHM={fwhm_fit:.2f}, sigma² (XAFS Input)={sigma2_fit:.4f}"
        )
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()

        # Return everything needed for display or downstream logic
        return {
            "figure": fig,
            "fwhm": fwhm_fit,
            "sigma2": sigma2_fit,
            "centers": centers,
            "motif_areas": motif_areas,
            "fit_sigma": sigma_fit,
        }

    def fit_histogram_with_weighted_gaussians(
        self,
        bond_or_angle,
        type_label="Bond Pair",
        min_val=None,
        max_val=None,
        bins=100,
        show=True,
        components="gaussians",  # "lines", "gaussians", or "none"
    ):
        """Fits the merged histogram with a sum of Gaussians centered at
        every representative value from all motifs within min_val and
        max_val, each weighted by motif counts from md_prior_weights.

        All Gaussians share the same FWHM.
        """
        # Collect all bond/angle values and their motif weights
        motif_vals = self.get_representative_motif_values(
            bond_or_angle, type_label
        )
        if not hasattr(self, "md_prior_weights_map"):
            self.load_md_prior_weights()
        centers = []
        weights = []
        for motif_label, vals in motif_vals.items():
            if "_motif_" in motif_label:
                struct, motif_num = motif_label.split("_motif_", 1)
                motif_key = f"motif_{motif_num}"
            else:
                struct, motif_key = motif_label, None
            count = (
                self.md_prior_weights_map.get((struct, motif_key), {}).get(
                    "count", 0
                )
                or 0
            )
            for v in vals:
                if (min_val is None or v >= min_val) and (
                    max_val is None or v <= max_val
                ):
                    centers.append(v)
                    weights.append(count)
        centers = np.array(centers)
        weights = np.array(weights)
        if len(centers) == 0:
            print("No motif centers in range!")
            return None

        # Prepare histogram data
        merged_filename = f"{bond_or_angle}_merged.csv"
        values = np.array([v for _, v in self.merged_data[merged_filename]])
        if min_val is not None:
            values = values[values >= min_val]
        if max_val is not None:
            values = values[values <= max_val]
        counts, edges = np.histogram(
            values, bins=bins, range=(min_val, max_val)
        )
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Normalize motif weights to match total histogram area
        total_area = counts.sum() * (edges[1] - edges[0])
        motif_areas = weights / weights.sum() * total_area

        # Define shared-FWHM sum-of-Gaussians model
        def gauss(x, mu, sigma, area):
            return (
                area
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            )

        def sum_of_gaussians(x, sigma):
            return np.sum(
                [
                    gauss(x, mu, sigma, A)
                    for mu, A in zip(centers, motif_areas)
                ],
                axis=0,
            )

        # Initial guess: sigma is histogram std/2
        sigma0 = np.std(values) / 2 if len(values) > 1 else 0.2
        try:
            popt, pcov = curve_fit(
                lambda x, sigma: sum_of_gaussians(x, sigma),
                bin_centers,
                counts,
                p0=[sigma0],
                bounds=([0.001], [10]),
            )
        except Exception as e:
            print("Fit failed:", e)
            return None

        sigma_fit = popt[0]
        fwhm_fit = 2.355 * sigma_fit
        sigma2_fit = sigma_fit**2

        if show:
            plt.figure(figsize=(6, 4))
            plt.hist(
                values,
                bins=bins,
                range=(min_val, max_val),
                alpha=0.5,
                color="tab:blue",
                label="Merged histogram",
            )
            xfit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
            plt.plot(
                xfit,
                sum_of_gaussians(xfit, sigma_fit),
                "k-",
                label="Weighted Gaussian sum fit",
            )

            if components == "lines":
                for mu in centers:
                    plt.axvline(mu, color="gray", lw=1, alpha=0.5)
            elif components == "gaussians":
                for i, (mu, area) in enumerate(zip(centers, motif_areas)):
                    plt.plot(
                        xfit,
                        gauss(xfit, mu, sigma_fit, area),
                        "--",
                        lw=1.5,
                        alpha=0.7,
                        label=f"G{i+1}",
                    )
            # (components == "none": do nothing extra)

            plt.xlabel(
                "Bond Length (Å)"
                if type_label == "Bond Pair"
                else "Angle (deg)"
            )
            plt.ylabel("Count")
            # plt.legend()
            plt.title(
                f"Fit: {bond_or_angle}, FWHM={fwhm_fit:.2f}, sigma² (XAFS Input)={sigma2_fit:.4f}"
            )
            plt.tight_layout()
            plt.show()

        print(f"Fitted sigma^2: {sigma2_fit:.6f}")

        # Return all quantities needed for saving/reporting
        return fwhm_fit, sigma2_fit, centers, motif_areas, bin_centers, counts

    def save_weighted_gaussian_fit_report(
        self,
        bond_or_angle,
        fit_result,  # unused here but kept for compatibility
        bin_centers: np.ndarray,
        hist: np.ndarray,
        motif_areas: np.ndarray,
        centers: np.ndarray,
        sigma: float,
        save_folder: str,
        output_filename: str,
    ):
        """Save fit summary, histogram, components, and advanced fit
        statistics.

        Computes:
        - Chi-square (sum of (y - ŷ)²)
        - Reduced chi-square (chi² / (n - p))
        - R² (coefficient of determination)
        - RMSE
        - AIC & BIC
        """
        import math
        import os

        import numpy as np

        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, output_filename)

        # 1) Build the model and component curves
        def gauss(x, mu, sigma, area):
            return (
                area
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            )

        total_fit = sum(
            gauss(bin_centers, mu, sigma, area)
            for mu, area in zip(centers, motif_areas)
        )
        components = [
            gauss(bin_centers, mu, sigma, area)
            for mu, area in zip(centers, motif_areas)
        ]

        # 2) Residuals and sum-of-squares
        residuals = hist - total_fit
        SSE = np.sum(residuals**2)  # Sum Squared Error
        mean_hist = np.mean(hist)
        SST = np.sum((hist - mean_hist) ** 2)  # Total Sum of Squares
        SSR = np.sum((total_fit - mean_hist) ** 2)  # Explained Sum of Squares

        # 3) Degrees of freedom
        n = len(hist)
        p = len(centers) + 1  # one sigma + one area per center

        # 4) Metrics
        chi2 = SSE
        red_chi2 = chi2 / (n - p) if n > p else float("nan")
        R2 = SSR / SST if SST > 0 else float("nan")
        RMSE = math.sqrt(SSE / n)
        AIC = 2 * p + n * math.log(SSE / n)
        BIC = p * math.log(n) + n * math.log(SSE / n)

        # 5) Write report
        with open(save_path, "w") as f:
            f.write(f"# Fit report for: {bond_or_angle}\n")
            f.write(
                f"# Fit range: {min(bin_centers):.6f} to {max(bin_centers):.6f}\n"
            )
            f.write(f"# sigma (XAFS Input)      = {sigma:.6f}\n")
            f.write(f"# sigma² (XAFS Input)     = {sigma**2:.6f}\n")
            f.write(f"# Number of components    = {len(centers)}\n")
            f.write(f"# FWHM (XAFS Input)       = {2.355 * sigma:.6f}\n")
            f.write(f"# Number of bins: {len(bin_centers)}\n")
            f.write("\n# Motif centers and areas:\n")
            for i, (mu, area) in enumerate(zip(centers, motif_areas), start=1):
                f.write(f"#   G{i:<2} mu={mu:.6f}  area={area:.4f}\n")
            f.write("\n# Fit statistics:\n")
            f.write(f"#   Chi-square        = {chi2:.6f}\n")
            f.write(f"#   Reduced chi-square= {red_chi2:.6f}\n")
            f.write(f"#   R²                = {R2:.6f}\n")
            f.write(f"#   RMSE              = {RMSE:.6f}\n")
            f.write(f"#   AIC               = {AIC:.3f}\n")
            f.write(f"#   BIC               = {BIC:.3f}\n")
            f.write(
                "\n# Columns: bin_center  hist  fit_total  "
                + "  ".join(f"G{i}" for i in range(1, len(components) + 1))
                + "\n"
            )
            for idx in range(n):
                row = [
                    f"{bin_centers[idx]:.6f}",
                    f"{hist[idx]:.6f}",
                    f"{total_fit[idx]:.6f}",
                ]
                row += [f"{comp[idx]:.6f}" for comp in components]
                f.write("  ".join(row) + "\n")

        print(f"Saved fit results to {save_path}")
