<p align="center">
  <img src="https://raw.githubusercontent.com/kewh5868/SAXSShell/main/docs/source/img/saxshell_icon.png" alt="SAXSShell icon" width="180">
</p>

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

If you want an isolated environment first:

```bash
conda create -n saxshell-py312 python=3.12
```

```bash
conda run --no-capture-output -n saxshell-py312 python -m pip install saxshell
```

For editable local development:

```bash
conda run --no-capture-output -n saxshell-py312 python -m pip install -e .
```

You can also launch the code directly from a source checkout without installing
entry points:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs --help
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs ui
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

The installed umbrella entry point is:

```bash
conda run --no-capture-output -n saxshell-py312 saxshell --help
conda run --no-capture-output -n saxshell-py312 saxshell saxs --help
```

Standalone tools that install directly include:

- `bondanalysis`
- `blenderxyz`
- `clusterdynamics`
- `clusterdynamicsml`
- `clusters`
- `mdtrajectory`
- `pdfsetup`
- `saxshell`
- `xyz2pdb`

The SAXS and fullrmc interfaces are currently reached through the umbrella
command:

```bash
saxshell saxs ui
saxshell fullrmc ui /path/to/project
```

From a source checkout, the equivalent module launches are:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs ui
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.fullrmc ui /path/to/project
```
