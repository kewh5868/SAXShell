<p align="center">
  <img src="https://raw.githubusercontent.com/kewh5868/SAXSShell/main/docs/source/img/saxshell_icon.png" alt="SAXSShell icon" width="180">
</p>

# SAXSShell

SAXSShell is a Python toolkit for simulation-driven scattering workflows. It
combines Qt applications, supporting tools, and reusable Python workflows for:

- trajectory inspection and frame export
- XYZ to PDB conversion using reference molecules
- cluster extraction and bond-analysis pipelines
- SAXS project setup, prefit modeling, and pyDREAM fitting
- downstream fullrmc preparation helpers

## Documentation

Project documentation is published at:

- https://kewh5868.github.io/SAXSShell/

The documentation is organized by workflow rather than by source file. First
learn how to process your molecular dynamics trajectory into frames and
clusters, then create a project folder for the SAXSShell session:

- [Getting Started](https://kewh5868.github.io/SAXSShell/getting-started/installation/)
- [MD Extraction and Cluster Preparation](https://kewh5868.github.io/SAXSShell/user-guide/cluster-extraction/)
- [Quickstart](https://kewh5868.github.io/SAXSShell/getting-started/quickstart/)
- [XYZ to PDB Conversion](https://kewh5868.github.io/SAXSShell/user-guide/xyz2pdb-conversion/)
- [SAXS Prefit](https://kewh5868.github.io/SAXSShell/user-guide/saxs-prefit/)
- [pyDREAM Workflow](https://kewh5868.github.io/SAXSShell/user-guide/pydream-workflow/)
- [Template System](https://kewh5868.github.io/SAXSShell/user-guide/template-system/)

## Installation

SAXSShell is not pip-installable yet. Run it from a source checkout with the
repository conda environment file.

```bash
git clone https://github.com/kewh5868/SAXSShell.git
cd SAXSShell
conda env create -f requirements/saxshell-py312.yml
```

If the `saxshell-py312` environment already exists, update it from the same
file:

```bash
conda env update -n saxshell-py312 -f requirements/saxshell-py312.yml --prune
```

Launch the main SAXSShell application from the repository root:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs
```

## First Project

Start by preparing the simulation data that the SAXS project will consume:

1. Inspect the MD trajectory and export usable frames with `mdtrajectory`.
2. Convert frames with `xyz2pdb` only if residue-aware PDB files are needed.
3. Extract stoichiometry-sorted clusters with `clusters`.
4. Create a dedicated project folder for the SAXSShell session.
5. Open the SAXSShell application and choose that project folder in
   **Project Setup** before building SAXS components.

## Docs Local Preview

Install the pinned docs dependencies and start the local preview server from the
repository root:

```bash
conda run --no-capture-output -n saxshell-py312 python -m pip install -r requirements/docs.txt
conda run --no-capture-output -n saxshell-py312 mkdocs serve
```

Then open `http://127.0.0.1:8000/`.

## Standalone Tools

These supporting tools can be used independently from the main SAXSShell
window, or opened from the main UI when a project-backed workflow is needed:

- `mdtrajectory`: inspect MD trajectories, review optional CP2K energy data,
  choose an equilibration cutoff, and export selected frames.
- `xyz2pdb`: convert extracted XYZ frames into residue-aware PDB files with
  reference molecule definitions.
- `clusters`: extract stoichiometry-sorted cluster folders from exported XYZ
  or PDB frames.
- `bondanalysis`: measure bond-pair and angle distributions from cluster
  folders.
- `clusterdynamics`: build time-binned cluster population heatmaps and lifetime
  summaries.
- `clusterdynamicsml`: extend observed cluster dynamics with predicted larger
  structures and model-comparison outputs.
- `pdfsetup`: run Debyer-backed trajectory-averaged PDF and partial-PDF
  calculations.
- `blenderxyz`: render publication-style structure images with Blender.
- `representativefinder`: select representative structures from project-backed
  stoichiometry folders.
- `structureviewer`: inspect individual structure files in the SAXSShell
  structure viewer.

## External Applications

The conda environment file installs the Python stack, but several optional
SAXSShell applications call external software that must be installed separately:

| External software | Required by                                           | Install / docs                                                                                                                                                           |
| ----------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Debyer            | `pdfsetup` PDF and partial-PDF calculations           | [Debyer docs](https://debyer.readthedocs.io/en/latest/) and [Debyer GitHub](https://github.com/wojdyr/debyer)                                                            |
| Blender           | `blenderxyz` structure rendering                      | [Blender download](https://www.blender.org/download/) and [Blender installation manual](https://docs.blender.org/manual/en/latest/getting_started/installing/index.html) |
| Packmol           | `fullrmc` Packmol setup and solvent packing workflows | [Packmol GitHub](https://github.com/m3g/packmol) and [Packmol user guide](https://m3g.github.io/packmol/)                                                                |
| Docker            | `fullrmc` Packmol Docker link workflow                | [Get Docker](https://docs.docker.com/get-started/get-docker/)                                                                                                            |
