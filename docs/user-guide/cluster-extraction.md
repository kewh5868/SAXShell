# MD Extraction and Cluster Preparation

Cluster extraction is the bridge between raw trajectory data and the SAXS model.
In this repository, that bridge spans more than one tool.

## Typical path

1. Use `mdtrajectory` to inspect a trajectory and export frames.
2. Optionally use `xyz2pdb` if you need molecule-aware PDB frames.
3. Use `clusters` to extract stoichiometry-sorted cluster files.
4. Use `clusterdynamics` to build time-dependent cluster-distribution heatmaps
   and lifetime tables from the extracted frames.
5. Optionally use `bondanalysis` to measure bond or angle distributions on
   those clusters.
6. Optionally use **Debye-Waller Analysis** to estimate project-backed
   pairwise disorder coefficients from sorted PDB cluster folders.
7. Feed the resulting cluster folder into the SAXS project.

## `mdtrajectory`

This tool is responsible for:

- inspecting trajectory metadata
- optionally reading CP2K `.ener` files
- suggesting a cutoff
- exporting selected frames into a sibling folder
- writing `mdtrajectory_export.json` metadata beside the exported frames so
  downstream tools can recover the original frame indices and times

Example:

```bash
mdtrajectory inspect traj.xyz --energy-file traj.ener
mdtrajectory suggest-cutoff traj.xyz --energy-file traj.ener --temp-target-k 300 --window 3
mdtrajectory export traj.xyz --energy-file traj.ener --use-suggested-cutoff --temp-target-k 300 --window 3
```

When a cutoff is applied, the default folder name now uses the form
`splitxyz_f995_t497p5fs` or `splitpdb_f995_t497p5fs`, where `f995` records the
first exported source-frame index and `t497p5fs` records the first exported
time in femtoseconds.

## `xyz2pdb`

Use this only when residue identity matters downstream.

The current UI and CLI support:

- analyzing one sample frame and detecting the element inventory
- defining free atoms and reference molecules directly in the UI
- editing per-bond percentage tolerances and tight/relaxed search windows
- estimating molecule counts before export
- converting frames in the background while reusing the first-frame mapping template
- optional assertion mode for per-molecule geometry checks and reference updates
- browsing and creating reference molecules in the library

See the dedicated guide for the full interface and workflow:

- [XYZ to PDB Conversion](xyz2pdb-conversion.md)

## `clusters`

The cluster workflow supports both UI and CLI usage. Its CLI exposes separate
`inspect`, `preview`, and `export` modes, plus settings for:

- node, linker, and shell rules
- pair cutoffs
- box dimensions
- periodic boundary conditions
- search mode
- save-state frequency

The CLI help text explicitly calls out faster neighbor search modes such as
`kdtree` and `vectorized`.

## `clusterdynamics`

This application consumes the extracted XYZ or PDB frames from `mdtrajectory`
and applies the same cluster definitions and pair-cutoff rules used by
`clusters`, but bins the results over time instead of writing one
stoichiometry-folder export.

Key outputs:

- time-binned cluster-distribution heatmaps
- optional CP2K `.ener` overlays aligned to the same time axis
- a sortable lifetime table by stoichiometry label
- saved JSON/CSV datasets that can be reopened later for plotting

## Downstream structure analysis

After the sorted cluster folders have been produced, the main UI exposes two
structure-analysis tools that reuse them:

- [Bond Analysis](bond-analysis.md) for bond-pair and angle-triplet
  distributions
- [Debye-Waller Analysis](debye-waller-analysis.md) for pairwise
  thermal-displacement coefficients from sorted `PDB` cluster folders

## What SAXS expects from this stage

The SAXS workflow expects a usable cluster folder with representative structure
files and enough component identity to build prior weights and scattering
components.

If you are unsure whether your cluster folder is ready, start in
**Project Setup** and confirm that the project can discover the expected
clusters before moving on.

## Related pages

- [Project Setup](../getting-started/project-setup.md)
- [Bond Analysis](bond-analysis.md)
- [Debye-Waller Analysis](debye-waller-analysis.md)
- [Project Configuration](project-configuration.md)
- [Cluster Dynamics](cluster-dynamics.md)
- [Cluster Dynamics ML](cluster-dynamics-ml.md)
- [SAXS Prefit](saxs-prefit.md)
