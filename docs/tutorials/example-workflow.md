# Example Workflow

This walkthrough shows a realistic high-level sequence without assuming a
specific chemistry beyond "simulation frames eventually become a SAXS project."

Run the commands from the repository root after creating the `saxshell-py312`
conda environment from `requirements/saxshell-py312.yml`.

## Step 1: inspect the trajectory

Start with the trajectory tool to confirm that the input is readable and, if
available, that the accompanying energy file is usable for cutoff analysis.

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory inspect traj.xyz --energy-file traj.ener
```

## Step 2: export usable frames

Use either a manual cutoff or the suggested one:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory export traj.xyz --energy-file traj.ener --use-suggested-cutoff --temp-target-k 300 --window 3
```

## Step 3: convert to PDB only if needed

If downstream logic needs molecule identity, convert the exported XYZ frames:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.xyz2pdb export splitxyz --config residue_map.json
```

## Step 4: extract clusters

Use the cluster workflow on the exported frame folder:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.cluster preview splitxyz
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.cluster export splitxyz
```

## Step 5: inspect distributions

Run bond analysis on the resulting cluster folder if you need bond-pair or
angle summaries:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.bondanalysis run clusters_splitxyz0001
```

## Step 6: build a SAXS project

Create a dedicated project folder, open the SAXS UI, and configure the project:

```bash
mkdir -p my_saxshell_project
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs
```

In the UI:

1. choose or create the project folder
2. select the experimental data
3. select the cluster folder
4. choose the template
5. create the computed distribution and build the project inputs

## Step 7: refine the Prefit model

Move to **SAXS Prefit**, inspect the parameter table, and compute geometry
metadata if the selected template requires it.

## Step 8: launch DREAM if needed

Only after Prefit looks reasonable should you move to **SAXS DREAM Fit** and
write the runtime bundle.

## Result

At the end of this path, you should have a reusable project directory, a Prefit
state that explains the current model, and optional DREAM artifacts for a more
formal fit.
