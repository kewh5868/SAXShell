# Quickstart

This quickstart starts where a new SAXSShell project usually starts: with a
molecular dynamics trajectory that needs to become a reusable SAXS project
folder.

In plain language, the goal is to turn a trajectory into exported frames,
clusters, and a dedicated SAXSShell project directory, then compare those
simulation-derived structures against experimental SAXS data.

Run commands from the repository root after creating the conda environment in
[Installation](installation.md).

## Prepare the MD trajectory first

Begin by confirming that the trajectory is readable and exporting the frame set
that downstream tools should use:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory inspect traj.xyz --energy-file traj.ener
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory suggest-cutoff traj.xyz --energy-file traj.ener --temp-target-k 300 --window 3
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory export traj.xyz --energy-file traj.ener --use-suggested-cutoff --temp-target-k 300 --window 3
```

If residue identity matters for your downstream analysis, convert the exported
XYZ frames before clustering:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.xyz2pdb export splitxyz --config residue_map.json
```

Extract the cluster folder that the SAXS project will consume:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.cluster inspect splitxyz
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.cluster export splitxyz
```

Create a dedicated project folder for the SAXSShell session. Keep the project
folder separate from raw trajectory output so saved SAXS state, computed
distributions, fit results, and optional project-backed calculations stay
together.

```bash
mkdir -p my_saxshell_project
```

The fastest way to understand the SAXS UI is to treat the first three tabs as a
sequence:

- **Project Setup** defines the project inputs and creates a saved computed
  distribution for one modeling branch.
- **SAXS Prefit** lets you inspect whether the chosen template and built
  components produce a sensible model preview.
- **SAXS DREAM Fit** takes the Prefit state and runs Bayesian sampling when you
  want a posterior distribution instead of just a single editable preview.

## Start in Project Setup

Launch the SAXS UI:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs
```

### What to do first

1. Create a new project directory or open an existing SAXS project.
2. Select the experimental SAXS dataset.
3. Select the cluster folder you want to model.
4. Optionally select the solvent SAXS dataset if the workflow needs it.
5. Choose the template, q-range, grid behavior, and excluded elements.
6. Choose the **Component build mode** for the modeling branch you want to
   save.
7. Click **Create Computed Distribution**.
8. Click **Build SAXS Components**.

!!! info "Image placeholder"
Add a screenshot of the **Project Setup** tab after a project is loaded,
with the project path, data selectors, computed-distribution controls, and
component-build controls visible.

### About computed distributions

A computed distribution is SAXSShell's saved record of one Project Setup branch.
In practice it captures the active template, cluster source, q-range choices,
component-build mode, and related settings that define how SAXS components
should be generated for that branch.

### Debye-Waller note

!!! warning "Debye-Waller status"
**Compute Debye-Waller Factors (beta)** is currently in testing and has a
known bug. Treat that path as provisional, and verify any saved outputs
before you rely on them in later workflows.

## Move to SAXS Prefit

After components exist for the active computed distribution, move to
**SAXS Prefit**.

This is the tab where you answer practical questions such as:

- does the current template behave sensibly against the experimental trace
- do the built components look reasonable
- do any geometry-aware templates need cluster geometry metadata before the
  model can update
- which parameters should stay fixed, vary, or be expressed through simple
  relationships

!!! info "Image placeholder"
Add a screenshot of the **SAXS Prefit** tab showing the main plot, the
parameter table, and any geometry or solution-estimator controls that
should be called out to a first-time user.

## Use SAXS DREAM Fit when Prefit is stable

Only move to **SAXS DREAM Fit** after Prefit looks reasonable.

The DREAM tab uses the current Prefit state to prepare a pyDREAM runtime bundle
and then sample plausible parameter combinations. Use it when you want
uncertainty estimates, posterior summaries, or a more formal Bayesian fit.

!!! info "Image placeholder"
Add a screenshot of the **SAXS DREAM Fit** tab showing the parameter map,
runtime settings, and result-preview area that a new user should inspect
first.

## Optional upstream analysis

The SAXS tabs assume you already have a usable project input set. Depending on
your question, you may also want to analyze the prepared clusters before
building SAXS components:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.bondanalysis run clusters_splitxyz0001
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.clusterdynamics splitxyz --project-dir my_saxshell_project
```

## Next steps

- Go to [Project Setup](project-setup.md) for a more detailed setup sequence.
- Use [GUI Overview](../user-guide/gui-overview.md) if you want the main window
  mapped out before exploring the deeper user-guide pages.
- Use [SAXS Prefit](../user-guide/saxs-prefit.md) and
  [pyDREAM Workflow](../user-guide/pydream-workflow.md) once your computed
  distribution is in place.
