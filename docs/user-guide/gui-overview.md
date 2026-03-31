# GUI Overview

SAXSShell is not a single-window application. The repository contains multiple
Qt workflows, each with its own UI and CLI entry point.

Most tools install as direct top-level commands. The SAXS and fullrmc
applications currently route through the umbrella `saxshell` CLI, or through
`python -m` module execution when you are running from a source checkout.

## Main applications

### `mdtrajectory`

Use this when you need to inspect trajectories, choose an equilibration cutoff,
and export frames for downstream analysis.

### `xyz2pdb`

Use this when exported XYZ frames need molecule identification or residue-aware
PDB conversion before clustering.

### `clusters`

Use this when you need to build cluster-network exports from extracted frames.

### `clusterdynamics`

Use this when you need time-binned cluster-distribution heatmaps, optional
energy overlays, and lifetime / association / dissociation summaries from an
extracted XYZ or PDB frame folder.

### `clusterdynamicsml`

Use this when you want to extrapolate larger cluster candidates from the
observed smaller-cluster dynamics and structure library, generate predicted
structure files, and compare observed-only versus observed-plus-predicted SAXS
models.

### `bondanalysis`

Use this when you need bond-pair and angle-distribution measurements on the
cluster folders.

### `pdfsetup`

Use this when you need Debyer-backed trajectory-averaged PDF or partial-PDF
calculations, saved PDF calculation sets inside a SAXSShell project, and quick
switching between little `g(r)`, big `G(r)`, and `R(r)` representations.

### `saxshell saxs`

Use this for SAXS project management, prefit modeling, pyDREAM refinement, and
template-driven workflows.

### `saxshell fullrmc`

Use this when you want to prepare downstream fullrmc or Packmol-oriented
artifacts from a SAXS project.

## SAXS application tabs

The current SAXS UI exposes three primary tabs:

- **Project Setup**
- **SAXS Prefit**
- **SAXS DREAM Fit**

These tabs are not isolated. The active template, component list, geometry
metadata, and saved state all move between them.

## How the tabs relate

### Project Setup

Defines the project inputs and template choice.

### SAXS Prefit

Builds the lmfit-side preview around the current template, parameter table, and
cluster geometry metadata.

### SAXS DREAM Fit

Builds and runs the pyDREAM workflow once Prefit is in a usable state.

## Shared UI patterns

Several newer SAXS UI surfaces follow the same patterns:

- progress bars with text status
- long-running background tasks
- table-based editing for parameters or cluster geometry metadata
- plot control toggles for experimental, model, and solvent traces

## TODO

TODO: add screenshots once the docs site has a stable asset pipeline and the
UI labels settle after the current SAXS workflow changes.

??? note "Artwork Attribution"
The SAXSShell application icon used across the UI, documentation site, and
repository README pages is based on artwork generated with ChatGPT (OpenAI).
