# SAXSShell

SAXSShell is a workflow-oriented toolkit for turning molecular simulation output
into structural analysis and SAXS fitting workflows. The repository combines:

- Qt desktop applications for interactive use
- command-line interfaces for reproducible batch runs
- Python workflow classes that mirror the same operations for notebooks and
  scripts

The project is aimed at researchers who need to move from trajectory files to
cluster populations, distribution analysis, SAXS component building, prefit
screening, and Bayesian fitting without constantly switching tools.

## Who this documentation is for

This site is organized around tasks:

- users preparing a new SAXS project from cluster folders
- users running SAXS Prefit and pyDREAM refinement
- users maintaining or extending the template system
- contributors working on the codebase itself

If you want a quick entry point, start with:

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Project Setup](getting-started/project-setup.md)

## Main workflow

The current repo supports an end-to-end path that usually looks like this:

1. Inspect and split trajectories with `mdtrajectory`.
2. Optionally convert XYZ frames to residue-aware PDB files with `xyz2pdb`.
3. Extract clusters with `clusters`.
4. Analyze time-dependent cluster populations with `clusterdynamics`.
5. Measure bond and angle distributions with `bondanalysis`.
6. Build a SAXS project with `saxs`.
7. Refine the project in **SAXS Prefit** and, if needed, run **pyDREAM**.
8. Use the resulting distributions and selected structures in downstream tools
   such as `fullrmc`.

## Documentation map

### Getting started

Use this section if you are setting up the software for the first time or want
to create your first project quickly.

### User guide

Use this section when you already know the rough workflow and need details
about GUI tabs, file layout, template behavior, or export paths.

### Tutorials

Use this section for longer, task-based walkthroughs that connect several tools
in sequence.

### API

Use this section if you want the shortest route to the reusable workflow
classes.

### Development

Use this section if you are contributing code, working on CI, or changing the
documentation site itself.

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
