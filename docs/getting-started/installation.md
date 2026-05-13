# Installation

## Current install path

SAXSShell is not pip-installable yet. Run it from a source checkout and create
the pinned conda environment from the repository environment file.

Use Python 3.12 through the provided `saxshell-py312` environment. The
repository CI and local guidance are pinned around Python 3.12 because the
current Qt stack is not ready for Python 3.14.

## Clone the repository

```bash
git clone https://github.com/kewh5868/SAXSShell.git
cd SAXSShell
```

Run the rest of the commands from the repository root unless a page says
otherwise.

## Create the conda environment

Create the environment from the checked-in `.yml` file:

```bash
conda env create -f requirements/saxshell-py312.yml
```

### Windows users

On native Windows, create the environment from the Windows-specific environment
file:

```cmd
conda env create -f requirements\saxshell-py312-win.yml
```

If the environment already exists, update it from the same file:

```cmd
conda env update -n saxshell-py312 -f requirements\saxshell-py312-win.yml --prune
```

### Linux, macOS, and WSL users

If the environment already exists, update it from the default environment file:

```bash
conda env update -n saxshell-py312 -f requirements/saxshell-py312.yml --prune
```

The examples use `conda run --no-capture-output -n saxshell-py312 ...` so Qt
logs and tracebacks remain visible in the terminal.

## Launch SAXSShell

Start the main SAXSShell application from the repository root.

### Linux, macOS, and WSL users

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs
```

### Windows users

There are two supported launch methods depending on whether you are using
Anaconda Prompt or Windows PowerShell.

#### From Anaconda Prompt

Activate the environment, set `PYTHONPATH`, and launch the SAXS UI:

```cmd
conda activate saxshell-py312
set PYTHONPATH=src
python -m saxshell.saxs
```

You can also launch without activating the environment:

```cmd
set PYTHONPATH=src
conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs
```

#### From Windows PowerShell

Set `PYTHONPATH` using PowerShell syntax, then launch through `conda run`:

```powershell
$env:PYTHONPATH = "src"
conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs
```

If your PowerShell session is configured for conda activation, this also works:

```powershell
conda activate saxshell-py312
$env:PYTHONPATH = "src"
python -m saxshell.saxs
```

The application opens to the main SAXS workflow. Create or select a dedicated
project folder in **Project Setup** after your trajectory-derived frames and
clusters are ready.

## Recommended starting point

Before spending time in Prefit or DREAM, prepare the simulation data that the
SAXS project will consume:

1. Inspect the MD trajectory and export a frame folder with `mdtrajectory`.
2. Convert exported XYZ frames with `xyz2pdb` only if downstream analysis needs
   residue-aware PDB files.
3. Extract stoichiometry-sorted clusters with `clusters`.
4. Create a dedicated project folder for the SAXSShell session.
5. Launch SAXSShell and choose that project folder in **Project Setup**.

See [MD Extraction and Cluster Preparation](../user-guide/cluster-extraction.md)
for the trajectory-to-clusters path.

## Standalone tools

These supporting tools can be used independently from the main SAXSShell
window, or opened from the main UI when a project-backed workflow is needed:

| Tool                   | Short description                                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `mdtrajectory`         | Inspects MD trajectories, reads optional CP2K energy files, helps choose equilibration cutoffs, and exports selected frames. |
| `xyz2pdb`              | Converts extracted XYZ frames into residue-aware PDB files using reference molecule definitions.                             |
| `clusters`             | Extracts stoichiometry-sorted cluster folders from exported XYZ or PDB frames.                                               |
| `bondanalysis`         | Measures bond-pair and angle distributions from cluster folders.                                                             |
| `clusterdynamics`      | Builds time-binned cluster population heatmaps, energy overlays, and lifetime summaries.                                     |
| `clusterdynamicsml`    | Extends observed cluster dynamics with predicted larger structures and model-comparison outputs.                             |
| `pdfsetup`             | Runs Debyer-backed trajectory-averaged PDF and partial-PDF calculations.                                                     |
| `blenderxyz`           | Creates publication-style structure renders with Blender.                                                                    |
| `representativefinder` | Selects representative structures from project-backed stoichiometry folders.                                                 |
| `structureviewer`      | Opens individual structure files in the SAXSShell structure viewer.                                                          |

From the source checkout, these tools are reached through their Python modules
inside the `saxshell-py312` environment. For example:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.mdtrajectory inspect traj.xyz --energy-file traj.ener
```

## External application dependencies

The conda environment file installs the Python stack. Some optional
SAXSShell applications call external software that must be installed separately.

| External software | Required by                                           | Install / docs                                                                                                                                                           |
| ----------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Debyer            | `pdfsetup` PDF and partial-PDF calculations           | [Debyer docs](https://debyer.readthedocs.io/en/latest/) and [Debyer GitHub](https://github.com/wojdyr/debyer)                                                            |
| Blender           | `blenderxyz` structure rendering                      | [Blender download](https://www.blender.org/download/) and [Blender installation manual](https://docs.blender.org/manual/en/latest/getting_started/installing/index.html) |
| Packmol           | `fullrmc` Packmol setup and solvent packing workflows | [Packmol GitHub](https://github.com/m3g/packmol) and [Packmol user guide](https://m3g.github.io/packmol/)                                                                |
| Docker            | `fullrmc` Packmol Docker link workflow                | [Get Docker](https://docs.docker.com/get-started/get-docker/)                                                                                                            |

### Debyer

The `pdfsetup` application launches the `debyer` executable as an external
backend. SAXSShell does not bundle that binary. Install Debyer separately and
make sure `debyer` is available on `PATH` before running trajectory-averaged
PDF calculations.

Debyer's upstream repository documents source builds with `autoconf`,
`automake`, `gengetopt`, `./configure`, and `make`. Follow the current upstream
instructions for your platform.

### Blender

The `blenderxyz` renderer needs Blender installed separately. It works best
when `blender` is on `PATH`, but the renderer UI can also browse to a Blender
executable or `.app` bundle.

### Packmol and Docker

The `fullrmc` Packmol setup workflow needs Packmol when you are preparing packed
coordinate files. The Packmol Docker link workflow additionally needs Docker so
SAXSShell can validate and sync files into a Packmol-ready container.

Install Docker only if you plan to use the Packmol container workflow. If you
use Packmol outside Docker, make sure the selected workflow can reach the
`packmol` executable in that environment.

## Optional docs dependencies

If you want to preview this documentation site locally:

```bash
conda run --no-capture-output -n saxshell-py312 python -m pip install -r requirements/docs.txt
conda run --no-capture-output -n saxshell-py312 mkdocs serve
```
