# Installation

## Python version

Use Python 3.12 for the current supported path.

The repository CI and local conda guidance are pinned around Python 3.12
because the current PySide6 stack in this repo is not ready for Python 3.14.

## Create a conda environment

The current docs examples assume a Python 3.12 environment named
`saxshell-py312`:

```bash
conda create -n saxshell-py312 python=3.12
```

You can either activate that environment first or keep commands explicit with
`conda run --no-capture-output -n saxshell-py312 ...`. The
`--no-capture-output` flag is useful for Qt applications because terminal logs
and tracebacks stay visible.

## Install from PyPI

```bash
conda run --no-capture-output -n saxshell-py312 python -m pip install saxshell
```

After installation, confirm the umbrella CLI is available:

```bash
conda run --no-capture-output -n saxshell-py312 saxshell --help
```

## Install from source

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/kewh5868/SAXSShell.git
cd SAXSShell
conda run --no-capture-output -n saxshell-py312 python -m pip install -e .
```

This installs the package and the command-line entry points defined in the
project metadata.

## Run directly from a source checkout

If you want to launch the software from the repository root without installing
editable entry points yet, export `PYTHONPATH=src` and run the relevant module
inside the conda environment:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxshell --help
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs --help
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs ui
```

Common translations from installed CLI form to source-checkout form are:

- `mdtrajectory ...` -> `PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory ...`
- `clusters ...` -> `PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.cluster ...`
- `blenderxyz ...` -> `PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.toolbox.blender.cli ...`
- `saxshell saxs ...` -> `PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs ...`
- `saxshell fullrmc ...` -> `PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.fullrmc ...`

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
- `blenderxyz`
- `clusterdynamics`
- `clusterdynamicsml`
- `bondanalysis`
- `pdfsetup`
- `xyz2pdb`

The SAXS and fullrmc CLIs are currently reached through the umbrella command
rather than separate installed scripts:

```bash
saxshell saxs --help
saxshell fullrmc --help
```

## Installation notes

- The SAXS UI and several other tools use PySide6 for the GUI.
- The SAXS workflow also depends on scientific Python packages such as NumPy,
  SciPy, and lmfit.
- The SAXS Debye component builder uses `xraydb`.
- The `blenderxyz` application also requires a separate Blender installation:
  <https://www.blender.org/download/>
- The Blender renderer works best when `blender` is on `PATH`, but you can also
  browse to the Blender executable or `.app` bundle from inside the UI.

## Debyer installation for PDF calculations

The `pdfsetup` application uses
[Debyer](https://debyer.readthedocs.io/en/latest/) as an external backend. That
means SAXSShell does not bundle the Debyer binary itself. You need to install
Debyer separately and make sure the `debyer` executable is available on your
`PATH`.

Useful upstream links:

- Debyer documentation: <https://debyer.readthedocs.io/en/latest/>
- Debyer GitHub repository: <https://github.com/wojdyr/debyer>

Debyer's official project documentation describes a native build based on its
own C/C++ source tree and autotools-style setup. SAXSShell's Debyer integration
does **not** require a Fortran runtime from Debyer itself. If you are installing
Debyer from source, follow the current upstream instructions rather than
assuming a Fortran toolchain is needed.

When the PDF application starts, it runs a quick Debyer availability check by:

1. locating `debyer` on `PATH`
2. attempting a lightweight `debyer --help` subprocess call

If that startup check fails, the PDF UI reports that immediately so you can
resolve the Debyer installation or local execution permissions before launching
a long trajectory-average job.

## TODO

TODO: add a short platform-specific troubleshooting section once the current
conda packaging and GUI runtime guidance are finalized.
