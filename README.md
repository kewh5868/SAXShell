# SAXSShell

SAXSShell is a Python toolkit for simulation-driven scattering workflows. It
combines Qt applications, command-line tools, and reusable Python workflows for:

- trajectory inspection and frame export
- XYZ to PDB conversion using reference molecules
- cluster extraction and bond-analysis pipelines
- SAXS project setup, prefit modeling, and pyDREAM fitting
- downstream fullrmc preparation helpers

## Documentation

Project documentation is published at:

- https://kewh5868.github.io/SAXSShell/

The documentation is organized by workflow rather than by source file, so the
best starting points are:

- [Getting Started](https://kewh5868.github.io/SAXSShell/getting-started/installation/)
- [SAXS Prefit](https://kewh5868.github.io/SAXSShell/user-guide/saxs-prefit/)
- [pyDREAM Workflow](https://kewh5868.github.io/SAXSShell/user-guide/pydream-workflow/)
- [Template System](https://kewh5868.github.io/SAXSShell/user-guide/template-system/)

## Installation

Use Python 3.12 for the smoothest experience with the current Qt stack.

```bash
python -m pip install saxshell
```

For editable local development:

```bash
python -m pip install -e .
```

## Docs Local Preview

Install the pinned docs dependencies and start the local preview server from the
repository root:

```bash
python -m pip install -r requirements/docs.txt
mkdocs serve
```

Then open `http://127.0.0.1:8000/`.

## CLI Entry Points

The umbrella entry point is:

```bash
saxshell --help
```

Common sub-tools also install directly:

- `mdtrajectory`
- `clusters`
- `bondanalysis`
- `xyz2pdb`
- `saxs`
- `fullrmc`
