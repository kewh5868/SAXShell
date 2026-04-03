# Cluster Dynamics

`clusterdynamics` is the time-resolved companion to `clusters`. It analyzes the
same extracted XYZ or PDB frame folders from `mdtrajectory`, reuses the same
cluster-definition and pair-cutoff logic, and converts the per-frame cluster
counts into:

- a time-binned cluster-distribution heatmap
- an optional lower subplot from a CP2K `.ener` file
- a sortable lifetime table by stoichiometry label
- reloadable JSON/CSV datasets for later plotting

## Inputs

The application expects:

- an extracted frame folder produced by `mdtrajectory`
- optional CP2K `.ener` data for the lower overlay subplot
- the same atom-type definitions, pair cutoffs, shell options, and PBC/search
  settings you would use in `clusters`

The left pane includes all of the cluster-rule controls from the cluster
extraction UI, plus the time-axis controls and dataset save/load actions.

## Time Axis Rules

The time axis is resolved in this order:

1. `mdtrajectory_export.json` if the frame folder came from a recent
   `mdtrajectory` export. This is the most reliable source because it stores
   the original frame indices and frame times written during export.
2. `frame_<index>.xyz` or `frame_<index>.pdb` filenames multiplied by the
   user-specified frame timestep.
3. A sequential fallback that starts from the folder/start-time field.

The frame timestep field defaults to `0.5 fs`.

The heatmap binning now uses an integer `frames / colormap timestep` control.
The UI shows the derived `colormap timestep used (fs)` field so the effective
heatmap timestep is always an exact multiple of the frame timestep.

`mdtrajectory` cutoff exports now use folder names such as
`splitxyz_f995_t497p5fs`. The `f995` part records the first exported
source-frame index, and `t497p5fs` records the first exported time in
femtoseconds. The `t...fs` portion is auto-filled into the folder/start-time
field when it is detected.

### Important example

If the folder is `splitxyz_f995_t497p5fs`, the frame timestep is `0.5 fs`, and
the first file is `frame_0995.xyz`, the resolved frame time is:

```text
995 x 0.5 fs = 497.5 fs
```

If an older folder such as `splitxyz_f847fs` is loaded, the preview still reads
that legacy cutoff/start-time tag. When the folder tag and the resolved frame
times disagree, the heatmap and lifetime calculations follow the resolved frame
times, not the folder label.

## Typical UI Workflow

1. Load the extracted XYZ/PDB frames folder.
2. Optionally load a CP2K `.ener` file.
3. Confirm the active SAXSShell project if you want saved datasets to default
   into `exported_results/data/clusterdynamics`.
4. Enter the cluster definitions, pair cutoffs, shell options, and PBC/search
   settings.
5. Confirm the frame timestep, frames per colormap timestep, derived
   colormap timestep, and analysis start/stop window.
6. Run the analysis.
7. Adjust the heatmap display mode, time units, colormap, quantile limits, and
   optional overlay interactively.
8. Use the **Saved Results** panel to save the current result or reopen a
   previously saved dataset without rerunning the frame analysis.

If the tool is launched from the main SAXS UI, it inherits the active project
directory automatically.

## Saved Outputs

The save action writes a JSON dataset plus companion CSV files beside it:

- `*_cluster_distribution.csv`
- `*_lifetime.csv`
- `*_energy.csv` when an energy overlay is present

The JSON file is the reloadable artifact. It stores the plotted matrices,
summary tables, time-axis metadata, and optional energy data needed to reopen
the analysis. In the UI, these controls live in the **Saved Results** panel so
they are separate from the **Run Analysis** controls.

## Heatmap Data

Each heatmap row is a stoichiometry label, and the full y-axis spans the labels
observed across all time bins in the current analysis window.

The display mode can be switched interactively between:

- raw counts per bin
- fraction of all clusters in the bin
- mean count per sampled frame in the bin

The color scaling uses quantile limits instead of fixed min/max values so large
outliers do not flatten the rest of the heatmap.

## Association, Dissociation, and Lifetimes

The lifetime table is computed from the per-label count series over consecutive
sampled frames.

- Association events: every positive increase in the count for a label between
  two adjacent sampled frames.
- Dissociation events: every negative decrease in that count between adjacent
  sampled frames.
- Association rate: `association_events / observation_window_ps`
- Dissociation rate: `dissociation_events / observation_window_ps`

This is a count-based kinetic summary. It does not claim atom-by-atom identity
tracking across frames. Instead, it reports how the occupancy of each
stoichiometry label changes between samples.

### Lifetime definition

- Completed lifetime: a cluster instance that appears after the first sampled
  frame and disappears before the end of the observation window.
- Window-truncated lifetime: a cluster instance that was already present in the
  first sampled frame or was still present in the last sampled frame.

Mean and standard deviation lifetimes are computed from completed lifetimes
only. The window-truncated count is reported separately so you can see how much
of the series is clipped by the analysis boundaries.

## Lifetime Table Columns

- `Label`: stoichiometry label, such as `Pb2I`
- `Size`: total number of atoms represented by the stoichiometry label
- `Mean lifetime (fs)`: average duration of completed lifetimes for that label
- `Std lifetime (fs)`: standard deviation of completed lifetimes
- `Completed`: number of completed lifetimes used in the mean/std calculation
- `Window-truncated`: number of lifetimes clipped by the start or end of the
  sampled window
- `Assoc. rate (1/ps)`: positive count changes per picosecond over the selected
  analysis window
- `Dissoc. rate (1/ps)`: negative count changes per picosecond over the
  selected analysis window
- `Occupancy (%)`: fraction of sampled frames in which at least one cluster of
  that label was present
- `Mean count/frame`: average number of clusters with that label per sampled
  frame

Sort the `Lifetime` tab by the `Size` column if you want the older “lifetime by
size” view without a separate tab.

## Cluster-Distribution CSV Columns

The saved `*_cluster_distribution.csv` file stores one row per
`label x time-bin` combination with:

- the label and cluster size
- bin index, start, stop, and center in femtoseconds
- raw count in the bin
- fraction in the bin
- mean count per sampled frame in the bin
- number of sampled frames in that bin
- total clusters in that bin

## When Warnings Appear

The preview and summary can include warnings when:

- the folder/start-time tag such as `_t497p5fs` was not found
- `mdtrajectory_export.json` is missing or incomplete
- the folder/start-time tag and the resolved frame times disagree

These warnings are informational. The analysis still runs, but the UI makes the
time basis explicit so the heatmap and lifetime interpretation stay transparent.

## Related pages

- [Cluster Extraction](cluster-extraction.md)
- [Cluster Dynamics ML](cluster-dynamics-ml.md)
- [Project Configuration](project-configuration.md)
- [Results and Export](results-and-export.md)
