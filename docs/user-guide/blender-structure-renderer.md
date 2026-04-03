# Blender Structure Renderer

The Blender tool is SAXSShell's publication-rendering application for atomistic
structures. It is exposed as the standalone `blenderxyz` program and from the
main SAXSShell window through `Tools > Open Blender XYZ Renderer`.

## Purpose

Use the Blender renderer when you want structure figures that are cleaner and
more presentation-ready than the fast inline visualizer preview. The tool is
designed for:

- generating publication graphics from `.xyz` or `.pdb` structures
- previewing and batch-rendering multiple orientations from one structure
- matching a flatter visualizer-like cartoon look or brighter high-key figure styles
- saving transparent PNG outputs for papers, slides, and figure assembly
- optionally saving a separate atom legend image or a `.blend` scene per render

## Main capabilities

The current Blender renderer can:

- load an `XYZ` or `PDB` structure file
- auto-generate preset axis views and computed photoshoot-style orientations
- add, duplicate, and edit custom orientation rows
- override aesthetic, render quality, and lighting per orientation row
- preview the active orientation before rendering
- edit VESTA-style pair-specific bond thresholds for the active structure
- edit per-element atom colors and sizes and save named custom aesthetics across sessions
- remember bond-threshold overrides for a structure across sessions
- render transparent PNG outputs with orientation, style, and quality metadata in the filename

## Blender dependency

SAXSShell does not bundle Blender itself. Install Blender separately from the
official Blender download page:

- Blender download: <https://www.blender.org/download/>

After installation, either:

- make sure the `blender` executable is available on your `PATH`, or
- provide the Blender executable or `.app` bundle path in the renderer window
  or with `--blender-executable` when launching from the terminal

If Blender is not on `PATH`, the UI can still be used by browsing to the
Blender application manually.

## Running the application independently from the terminal

You do not need to open the main SAXSShell UI first. The Blender renderer can
be launched directly as its own Qt application.

### Installed package

If SAXSShell is installed into your environment, start the Blender tool with:

```bash
blenderxyz
```

You can also prefill a structure file or Blender location:

```bash
blenderxyz path/to/structure.xyz
blenderxyz path/to/structure.pdb --blender-executable /Applications/Blender.app
```

### Source checkout

From the repository root, launch the same standalone application with:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 \
  python -m saxshell.toolbox.blender.cli
```

You can also pass the optional structure file and Blender path in source mode:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 \
  python -m saxshell.toolbox.blender.cli path/to/structure.xyz \
  --blender-executable /Applications/Blender.app
```

## Typical workflow

1. Launch `blenderxyz`.
2. Choose an `XYZ` or `PDB` file.
3. Confirm or browse to the Blender executable if needed.
4. Review the generated orientation rows and duplicate or add custom rows.
5. Adjust lighting, aesthetic, and render quality on the rows you want.
6. Optionally edit bond thresholds or create a saved custom aesthetic.
7. Render the selected rows to a destination folder.

## Related pages

- [Installation](../getting-started/installation.md)
- [GUI Overview](gui-overview.md)
- [Results and Export](results-and-export.md)
