# Cluster Extraction

Cluster extraction is the bridge between raw trajectory data and the SAXS model.
In this repository, that bridge spans more than one tool.

## Typical path

1. Use `mdtrajectory` to inspect a trajectory and export frames.
2. Optionally use `xyz2pdb` if you need molecule-aware PDB frames.
3. Use `clusters` to extract stoichiometry-sorted cluster files.
4. Use `clusterdynamics` to build time-dependent cluster-distribution heatmaps
   and lifetime tables from the extracted frames.
5. Use `bondanalysis` to measure bond or angle distributions on those clusters.
6. Feed the resulting cluster folder into the SAXS project.

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
`splitxyz_f847fs` or `splitpdb_f847fs`, where the `f847fs` portion records the
cutoff time in femtoseconds.

## `xyz2pdb`

Use this only when residue identity matters downstream.

The current CLI can:

- inspect the input and reference library
- preview residue assignment
- export converted PDB files
- list or add reference molecules

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

See [Cluster Dynamics](cluster-dynamics.md) for the full workflow, timing
rules, and the definitions of the lifetime/rate columns.

## `bondanalysis`

Bond analysis is downstream of cluster extraction. Use it to derive bond-pair
and angle-triplet distributions from the stoichiometry folders produced by the
cluster workflow.

## What SAXS expects from this stage

The SAXS workflow expects a usable cluster folder with representative structure
files and enough component identity to build prior weights and scattering
components.

If you are unsure whether your cluster folder is ready, start in
**Project Setup** and confirm that the project can discover the expected
clusters before moving on.
