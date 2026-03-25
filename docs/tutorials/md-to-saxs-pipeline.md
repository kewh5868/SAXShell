# MD to SAXS Pipeline

This tutorial focuses on the command-line side of the pipeline so the overall
flow is easy to automate.

## Starting point

Assume you have:

- a trajectory file such as `traj.xyz`
- optionally a CP2K energy file such as `traj.ener`
- optionally a residue-mapping JSON file for `xyz2pdb`

## Export frames

```bash
mdtrajectory inspect traj.xyz --energy-file traj.ener
mdtrajectory suggest-cutoff traj.xyz --energy-file traj.ener --temp-target-k 300 --window 3
mdtrajectory export traj.xyz --energy-file traj.ener --use-suggested-cutoff --temp-target-k 300 --window 3
```

## Convert to residue-aware PDB, if needed

```bash
xyz2pdb preview splitxyz --config residue_map.json
xyz2pdb export splitxyz --config residue_map.json
```

## Extract clusters

```bash
clusters inspect splitxyz
clusters export splitxyz --use-pbc
```

The exact node, linker, shell, and cutoff settings depend on the chemistry of
your system, so keep those definitions next to your project inputs rather than
hard-coding them into ad hoc notebooks.

## Analyze distributions

```bash
bondanalysis inspect clusters_splitxyz0001
bondanalysis run clusters_splitxyz0001
```

## Build the SAXS project

```bash
saxs ui
```

There is currently no one-shot CLI that replaces the full Project Setup tab, so
the usual next step is interactive project configuration in the SAXS UI.

## Prefit and DREAM

Once the project is configured:

- use **SAXS Prefit** for the lmfit-side model preview
- use **SAXS DREAM Fit** for posterior sampling

## TODO

TODO: add a chemistry-specific worked example once a stable example dataset is
checked into the repository or linked from releases.
