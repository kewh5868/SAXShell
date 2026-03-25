# GUI Overview

SAXShell is not a single-window application. The repository contains multiple
Qt workflows, each with its own UI and CLI entry point.

## Main applications

### `mdtrajectory`

Use this when you need to inspect trajectories, choose an equilibration cutoff,
and export frames for downstream analysis.

### `xyz2pdb`

Use this when exported XYZ frames need molecule identification or residue-aware
PDB conversion before clustering.

### `clusters`

Use this when you need to build cluster-network exports from extracted frames.

### `bondanalysis`

Use this when you need bond-pair and angle-distribution measurements on the
cluster folders.

### `saxs`

Use this for SAXS project management, prefit modeling, pyDREAM refinement, and
template-driven workflows.

### `fullrmc`

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
