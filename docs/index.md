# SAXSShell

![SAXSShell icon](source/img/saxshell_icon.svg){ width="220" }

SAXSShell is a workflow-oriented toolkit for comparing molecular dynamics
structures against small-angle X-ray scattering (`SAXS`) data.

In plain language, the main goal is to help you answer a solvation-structure
question: "which cluster populations, representative structures, and model
choices from my simulation best match the experimental SAXS signal?"

The repository combines:

- Qt desktop applications for interactive use
- source-checkout module launches for reproducible batch runs
- Python workflow classes that mirror the same operations for notebooks and
  scripts

The project is aimed at researchers who need to move from trajectory files to
cluster populations, distribution analysis, SAXS component building, Prefit
screening, and Bayesian fitting without constantly switching tools.

## Who this documentation is for

This site is organized around tasks:

- users preparing raw MD trajectories for a new SAXSShell project
- users preparing a new SAXS project from cluster folders
- users running SAXS Prefit and pyDREAM refinement
- users maintaining or extending the template system
- contributors working on the codebase itself

If you want a quick starting path, begin with installation, then learn how to
process your MD trajectories and create a project folder for the SAXSShell
session:

- [Installation](getting-started/installation.md)
- [MD Extraction and Cluster Preparation](user-guide/cluster-extraction.md)
- [Quickstart](getting-started/quickstart.md)
- [Project Setup](getting-started/project-setup.md)

## Main workflow

The current repo supports an end-to-end path that usually looks like this:

1. Inspect and split trajectories with `mdtrajectory`.
2. Optionally convert XYZ frames to residue-aware PDB files with `xyz2pdb`.
3. Extract clusters with `clusters`.
4. Analyze time-dependent cluster populations with `clusterdynamics`.
5. Optionally predict larger clusters and representative predicted structures
   with `clusterdynamicsml`.
6. Measure bond and angle distributions with `bondanalysis`.
7. Optionally compute project-backed representative structures with the full
   Representative Structures UI or the beta run-file based
   `representativefinder` source-module path.
8. Optionally compute project-backed Debye-Waller factors from sorted PDB
   cluster folders with **Debye-Waller Analysis**. This path is currently in
   testing and should be treated cautiously because the linked
   **Compute Debye-Waller Factors (beta)** workflow still has a known bug.
9. Optionally compute trajectory-averaged PDFs and partial PDFs with `pdfsetup`.
10. Create a dedicated SAXSShell project folder, launch the main application
    from the source checkout, create or load a computed distribution in
    **Project Setup**, and choose how SAXS components will be prepared.
11. Build components with one of the supported modes:
    `No Contrast (Debye)`, `Contrast (Debye)`,
    `1D Born Approximation (Average)`, or
    `3D FFT Born Approximation`.
12. Refine the project in **SAXS Prefit** and, if needed, run **pyDREAM**.
13. Use the resulting distributions and selected structures in downstream tools
    such as the `fullrmc` workflow.

## Documentation map

### Getting started

Use this section if you are setting up the software for the first time or want
to create your first project quickly.

### User guide

Use this section when you already know the rough workflow and need details
about GUI tabs, computed distributions, component-build modes, file layout,
template behavior, export paths, or the supporting applications that feed data
into the main SAXS UI.

The user guide is split into:

- **Main UI workflow elements** for the SAXSShell application and its
  project, computed distributions, Prefit, DREAM, template, and export
  behavior
- **Supporting applications** grouped the same way as the main `Tools` menu:
  `MD Extraction`, `Structure Analysis`, `Cluster Dynamics`, `PDF`,
  `Visualization`, `SAXS Calculation Preview`, `X-ray Toolkit`, and `(beta)`

### Tutorials

Use this section for longer, task-based walkthroughs that connect several tools
in sequence. These pages are still early drafts rather than complete tutorials.

### API

Use this section if you want the shortest route to the reusable workflow
classes. It is currently a provisional overview, not a complete API reference.

### Development

Use this section if you are contributing code, working on CI, or changing the
documentation site itself. These contributor notes are still being built out.

## Scope notes

The SAXS workflow in this repository is active and evolving. The documentation
focuses on features that are already represented in the current codebase. When
exact behavior is still evolving, the docs call that out explicitly instead of
guessing.

> Note
>
> SAXSShell is in the process of being renamed from `SAXShell`. During this
> transition, some internal package paths, CLI names, saved-settings namespaces,
> or compatibility references may still use `saxshell` or `SAXShell` even though
> the front-facing documentation and UI are being updated to `SAXSShell`.
