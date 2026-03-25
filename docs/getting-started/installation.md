# Installation

## Python version

Use Python 3.12 for the current supported path.

The repository CI and local conda guidance are pinned around Python 3.12
because the current PySide6 stack in this repo is not ready for Python 3.14.

## Install from PyPI

```bash
python -m pip install saxshell
```

After installation, confirm the umbrella CLI is available:

```bash
saxshell --help
```

## Install from source

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/kewh5868/SAXSShell.git
cd SAXSShell
python -m pip install -e .
```

This installs the package and the command-line entry points defined in the
project metadata.

## Optional docs dependencies

If you want to preview this documentation site locally:

```bash
python -m pip install -r requirements/docs.txt
mkdocs serve
```

## Installed commands

The current package exposes these top-level tools:

- `saxshell`
- `mdtrajectory`
- `clusters`
- `bondanalysis`
- `xyz2pdb`
- `saxs`
- `fullrmc`

## Installation notes

- `saxs` and several other tools use PySide6 for the GUI.
- `saxs` also depends on scientific Python packages such as NumPy, SciPy, and
  lmfit.
- The SAXS Debye component builder uses `xraydb`.

## TODO

TODO: add a short platform-specific troubleshooting section once the current
conda packaging and GUI runtime guidance are finalized.
