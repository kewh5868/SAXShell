# Blender Structure Renderer

`render_xyz_publication.py` renders an input XYZ or PDB structure as a polished
ball-and-stick image in Blender with:

- CPK atom colors
- VESTA-style bond searching with pair-specific minimum/maximum lengths
- tighter camera framing
- brighter studio lighting
- orientation-aware batch output filenames with style and quality metadata
- transparent PNG output
- an optional title label

## Basic usage

```bash
blender --background --python src/saxshell/toolbox/blender/render_xyz_publication.py -- \
  --input examples/molecule.xyz \
  --output-dir renders \
  --orientation isometric:35.264:0:45 \
  --orientation down_a_axis:0:0:90
```

## Batch launcher UI

You can also launch a Qt window that lets you choose an XYZ or PDB file, set the
destination folder, preview preset and computed orientations, add custom
Euler-angle orientations, duplicate rows for alternate looks, set the
aesthetic style, lighting level, and render quality directly on each
orientation row, choose whether a separate atom-legend PNG should be saved for
each row, and render the full selected set. The appearance controls include a
legend font selector. The left pane also includes a bond-threshold editor for
the active structure so you can adjust per-pair minimum/maximum bond lengths
before previewing or rendering. The render panel can save a `.blend` file for
each output so you can reopen and tweak the staged scene in Blender:

```bash
blenderxyz
```

Or from the main SAXSShell window: `Tools -> Open Blender XYZ Renderer`.

## Useful options

- `--width 2600 --height 2000`
  Controls output resolution.
- `--samples 256`
  Controls Cycles sampling. If omitted, the selected render-quality preset is used.
- `--lighting-level 4`
  Adjusts overall scene brightness from `1` (lowest) to `5` (brightest). The
  current baseline look is `2`.
- `--bond-color-mode neutral`
  Uses neutral gray bonds.
- `--bond-color-mode split`
  Splits each bond into two half-cylinders tinted by the connected atoms.
- `--bond-pair-threshold Cd:Se:0.0:3.15`
  Overrides the bond-search window for one element pair using VESTA-style
  minimum and maximum lengths in angstroms. Repeat as needed.
- `--orientation key:x:y:z[:atom_style:render_quality[:lighting_level]]`
  Adds an orientation in degrees for the batch render, with optional
  per-orientation style, quality, and lighting overrides.
- `--hide-title`
  Omits the text label.
- `--save-blend-files`
  Saves a `.blend` scene beside each rendered PNG.
- `--hide-hydrogen`
  Removes hydrogens from the rendered structure.
- `--input path/to/structure.pdb`
  PDB files are supported in addition to XYZ files.

## Extra styles

Additional flatter illustration-oriented presets are available in the UI and
CLI, including:

- `toon_matte`
- `poster_pop`
- `pastel_cartoon`
- `crystal_flat`
- `crystal_cartoon`
- `crystal_shadow_gloss`

For brighter publication-friendly results, start with `toon_matte`,
`poster_pop`, `soft_studio`, or `crystal_flat`. The cartoon presets are tuned
to track the flatter visualizer look more closely, and lighting levels `3` to
`4` are a good starting point when you want a high-key figure instead of a
low-light studio render.

Reference carbon-and-sulfur swatches for the UI can be regenerated with:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 \
  python src/saxshell/toolbox/blender/ui/generate_reference_atoms.py
```

## Notes

- Run this script from Blender, not from plain Python.
- The script clears the current Blender scene before building the render.
- The title defaults to the XYZ comment line or PDB title/header metadata,
  then the input filename stem if no structure comment is available.
- For multi-model PDB files, the first model is used.
- PNG outputs are always written with transparency.
- Atom legends generated from the UI are written as separate transparent PNG
  files beside the matching render.
- Batch outputs are written to the chosen folder with orientation, style, and
  render-quality metadata in the filename rather than a `_publication` suffix.
