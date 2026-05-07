# MD to SAXS Pipeline

This tutorial focuses on the command-line side of the pipeline so the overall
flow is easy to automate.

## Starting point

Assume you have:

- a trajectory file such as `traj.xyz`
- optionally a CP2K energy file such as `traj.ener`
- optionally a residue-mapping JSON file for `xyz2pdb`

Also assume you have cloned the repository, created the `saxshell-py312`
conda environment from `requirements/saxshell-py312.yml`, and are running these
commands from the repository root.

## Export frames

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory inspect traj.xyz --energy-file traj.ener
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory suggest-cutoff traj.xyz --energy-file traj.ener --temp-target-k 300 --window 3
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory export traj.xyz --energy-file traj.ener --use-suggested-cutoff --temp-target-k 300 --window 3
```

## Convert to residue-aware PDB, if needed

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.xyz2pdb preview splitxyz --config residue_map.json
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.xyz2pdb export splitxyz --config residue_map.json
```

## Extract clusters

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.cluster inspect splitxyz
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.cluster export splitxyz --use-pbc
```

The exact node, linker, shell, and cutoff settings depend on the chemistry of
your system, so keep those definitions next to your project inputs rather than
hard-coding them into ad hoc notebooks.

## Analyze distributions

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.bondanalysis inspect clusters_splitxyz0001
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.bondanalysis run clusters_splitxyz0001
```

## Build the SAXS project

```bash
mkdir -p my_saxshell_project
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs
```

The usual next step is interactive project configuration in the SAXS UI. Choose
the project folder you created, then select the experimental SAXS dataset and
the cluster folder produced above.

## Prefit and DREAM

Once the project is configured:

- use **SAXS Prefit** for the lmfit-side model preview
- use **SAXS DREAM Fit** for posterior sampling

## TODO

TODO: add a chemistry-specific worked example once a stable example dataset is
checked into the repository or linked from releases.
