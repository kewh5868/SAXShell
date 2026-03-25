# Cluster Extraction

Cluster extraction is the bridge between raw trajectory data and the SAXS model.
In this repository, that bridge spans more than one tool.

## Typical path

1. Use `mdtrajectory` to inspect a trajectory and export frames.
2. Optionally use `xyz2pdb` if you need molecule-aware PDB frames.
3. Use `clusters` to extract stoichiometry-sorted cluster files.
4. Use `bondanalysis` to measure bond or angle distributions on those clusters.
5. Feed the resulting cluster folder into the SAXS project.

## `mdtrajectory`

This tool is responsible for:

- inspecting trajectory metadata
- optionally reading CP2K `.ener` files
- suggesting a cutoff
- exporting selected frames into a sibling folder

Example:

```bash
mdtrajectory inspect traj.xyz --energy-file traj.ener
mdtrajectory suggest-cutoff traj.xyz --energy-file traj.ener --temp-target-k 300 --window 3
mdtrajectory export traj.xyz --energy-file traj.ener --use-suggested-cutoff --temp-target-k 300 --window 3
```

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
