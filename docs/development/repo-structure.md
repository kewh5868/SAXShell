# Repo Structure

This page is a short orientation guide for contributors.

## Top-level layout

```text
src/saxshell/
  bondanalysis/
  cluster/
  fullrmc/
  mdtrajectory/
  saxs/
  xyz2pdb/
tests/
requirements/
docs/
.github/workflows/
```

## Main application packages

### `src/saxshell/mdtrajectory`

Trajectory inspection, cutoff selection, frame export, and the matching Qt UI.

### `src/saxshell/xyz2pdb`

Residue-aware XYZ-to-PDB conversion, reference-library helpers, and UI code.

### `src/saxshell/cluster`

Cluster extraction workflows and UI.

### `src/saxshell/bondanalysis`

Bond-pair and angle-analysis workflows and UI.

### `src/saxshell/saxs`

SAXS-specific project management, templates, Debye profile generation, Prefit,
DREAM runtime support, and the main SAXS UI.

### `src/saxshell/fullrmc`

Downstream helpers and UI for preparing fullrmc-oriented artifacts from a SAXS
project.

## Tests

The repository keeps targeted tests in `tests/`, with separate files for SAXS,
UI, template-installation, fullrmc, xyz2pdb, and other workflows.

## Docs

The docs site is now a MkDocs project rooted at:

- `mkdocs.yml`
- `docs/`
- `requirements/docs.txt`

## Legacy and deprecated code

The repository still contains some `_deprecated` directories. These should not
be treated as the primary implementation path when adding new behavior.
