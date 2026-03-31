# Quickstart

This quickstart is intentionally practical. It is not the full workflow, but it
gets you from a trajectory or cluster folder to the relevant applications.

The command examples below assume the package is installed in your active
environment. If you are launching directly from a source checkout, use the
`PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m ...`
pattern from [Installation](installation.md). For example, `saxshell saxs ui`
maps to `python -m saxshell.saxs ui`.

## 1. Inspect and export frames

If you are starting from a trajectory:

```bash
mdtrajectory inspect traj.xyz --energy-file traj.ener
mdtrajectory export traj.xyz --energy-file traj.ener --use-suggested-cutoff --temp-target-k 300 --window 3
```

This gives you a folder of exported frames that can feed the next stage.

## 2. Optional XYZ to PDB conversion

If you need residue-aware PDB frames before clustering:

```bash
xyz2pdb preview splitxyz --config residue_map.json
xyz2pdb export splitxyz --config residue_map.json
```

Skip this stage if plain XYZ cluster extraction is enough for your system.

## 3. Extract clusters

Launch the cluster UI:

```bash
clusters
```

Or inspect a frame folder from the terminal:

```bash
clusters inspect splitxyz
```

## 4. Analyze bond and angle distributions

```bash
bondanalysis inspect clusters_splitxyz0001
bondanalysis run clusters_splitxyz0001
```

## 5. Launch the SAXS workflow

Open the SAXS UI:

```bash
saxshell saxs ui
```

Inside the UI, the normal path is:

1. Create or open a SAXS project.
2. Point the project at your experimental data and cluster folder.
3. Build the SAXS components in **Project Setup**.
4. Review the model in **SAXS Prefit**.
5. Run **SAXS DREAM Fit** if you need Bayesian refinement.

## 6. If you are starting from an existing project

You can open a project directly:

```bash
saxshell saxs ui /path/to/project
```

The same pattern also exists for the `fullrmc` UI:

```bash
saxshell fullrmc ui /path/to/project
```

## Next step

Go to [Project Setup](project-setup.md) for the first SAXS-specific workflow in
the GUI.
