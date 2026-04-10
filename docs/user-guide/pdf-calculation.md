# PDF Calculation

SAXSShell includes a Debyer-backed PDF application exposed as `pdfsetup` and as
`Tools > PDF > Open PDF Calculation` from the main SAXS UI.

When the tool is opened from the main UI, its saved calculations become part of
the same project-backed workflow state used by the SAXS tabs.

## What the application does

The PDF tool is designed for trajectory-average work, not only one-off single
structure calculations. It:

- inspects a folder of extracted `.xyz` or `.pdb` frames
- runs Debyer as a subprocess on each frame
- averages the total PDF/RDF/rPDF trace across the trajectory
- averages the Debyer partial pair traces for every element-element pair
- stores each computed calculation in the current SAXSShell project folder
- reloads older saved PDF calculations for comparison

By default, the UI plots the averaged little `g(r)` trace in black and keeps the
partial traces available as optional overlays.

## Debyer backend requirements

SAXSShell does not bundle Debyer. Install Debyer separately and make sure the
`debyer` executable can be launched from a terminal.

- Debyer docs: <https://debyer.readthedocs.io/en/latest/>
- Debyer GitHub: <https://github.com/wojdyr/debyer>

According to the upstream Debyer documentation and repository, Debyer is built
as its own native executable. The SAXSShell PDF application does **not** assume
that Debyer requires a Fortran runtime. Follow the current upstream Debyer
build/install instructions for your platform.

When the PDF UI opens, SAXSShell runs a quick runtime check:

1. locate `debyer` on `PATH`
2. attempt a short `debyer --help` subprocess call

If that check fails, the UI reports the failure immediately instead of waiting
until the full trajectory-average job is started.

## Main inputs

The left pane is intentionally close to the newer Cluster Dynamics ML layout.
The important settings are:

- **Project folder**
  Stores saved PDF calculations and reloadable metadata.
- **Frames folder**
  Folder containing either only `.xyz` frames or only `.pdb` frames.
- **Output prefix**
  Base name used for the saved calculation folder and output files.
- **Mode**
  Debyer mode. The default is `PDF`.
- **from / to / step**
  Radial grid settings. Defaults are `0.5`, `15`, and `0.01`.
- **Bounding box**
  The periodic box dimensions `a`, `b`, `c` in angstroms.
- **Atom count**
  Used to compute the number density `rho0`.
- **Store per-frame Debyer output files**
  Off by default. When off, SAXSShell keeps the averaged result and metadata but
  cleans up the temporary per-frame `.txt` files.
- **Solute elements**
  Optional comma-separated element list used to group partials into
  solute-solute, solute-solvent, and solvent-solvent families.

## Bounding box and number density

The Debyer UI tries to prefill the box and atom-count values before a run:

1. If the extracted-frame source metadata includes a sibling source filename
   with `_pbc_...` box tokens, SAXSShell uses that first.
2. Otherwise, it estimates the box from the coordinate extent of the first
   frame.
3. The user can still edit `a`, `b`, and `c` manually before running Debyer.

The number density is then computed as:

\[
\rho_0 = \frac{N}{V}
\]

where \(N\) is the atom count and \(V = abc\) for the orthorhombic box used by
the current Debyer integration.

## Averaging model

Each frame is processed independently by Debyer. SAXSShell then averages the
result column-by-column over the trajectory:

\[
\bar{y}_k(r) = \frac{1}{M}\sum_{m=1}^{M} y\_{k,m}(r)
\]

where:

- \(M\) is the number of frames
- \(k\) is either the total trace or one Debyer partial pair column
- \(y\_{k,m}(r)\) is the Debyer output for frame \(m\)

The saved averaged result therefore includes:

- the averaged total trace
- averaged element-pair partial traces
- metadata describing the raw Debyer mode, radial grid, frame count, box, and
  `rho0`

## Plot representations: `g(r)`, `G(r)`, and `R(r)`

The default plot mode is little \(g(r)\), but the UI can switch the loaded
result between three common total-scattering representations discussed in
Debyer's documentation and in the total-scattering formalism guide:

- Debyer docs: <https://debyer.readthedocs.io/en/latest/>
- Formalism guide:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7941302/>

For the SAXSShell PDF tool, the conversions are treated as:

\[
R(r) = 4\pi \rho_0 r^2 g(r)
\]

\[
G(r) = 4\pi \rho_0 r \left[g(r) - 1\right]
\]

and therefore:

\[
g(r) = \frac{R(r)}{4\pi \rho_0 r^2}
\]

\[
g(r) = 1 + \frac{G(r)}{4\pi \rho_0 r}
\]

This is why the box dimensions and atom count matter for the PDF application
even when the user ultimately wants to inspect `G(r)` or `R(r)`.

## Partial traces and grouped traces

Debyer can emit weighted partial pair traces as extra output columns. SAXSShell
stores the averaged pair traces and exposes them as optional overlays.

If the user provides **Solute elements**, the UI also groups those partials into
three high-level categories:

- `solute-solute`
- `solute-solvent`
- `solvent-solvent`

This gives a direct replacement path for older solvent/solute separation
workflows while keeping the saved Debyer result centered on the actual
element-pair partials.

## Saved calculation behavior

Each completed run is stored in the project folder under the exported Debyer
data area. When the PDF application is reopened with the same project:

- previously computed PDF calculations are listed in the saved-calculation box
- the most recent saved calculation is loaded automatically
- the plot and trace table update from the stored averaged data without
  rerunning Debyer

## UI layout

The current Debyer PDF UI is arranged as:

- **Left pane**
  Project settings, frame-folder inspection, Debyer settings, run button, and
  output console.
- **Right pane**
  Plot controls at the top, the main average/partial PDF plot in the middle,
  and a trace table at the bottom.

The trace table supports:

- per-trace visibility checkboxes
- color pickers
- separate toggles for the average trace, partial pair traces, and grouped
  traces

## Output files

Each saved calculation stores:

- a human-readable `calculation.json`
- the averaged Debyer output table
- optional per-frame Debyer `.txt` files when that option is enabled

That makes the saved calculation both:

- readable enough to inspect manually
- structured enough to support future replay or automated replotting features

## Current scope

The first Debyer integration focuses on trajectory-averaged total and partial
PDF traces. It does not yet replace every legacy notebook-era Debyer helper, but
it provides a stable project-backed path for repeated Debyer calculations inside
SAXSShell.

## Related pages

- [Project Configuration](project-configuration.md)
- [Results and Export](results-and-export.md)
- [GUI Overview](gui-overview.md)
