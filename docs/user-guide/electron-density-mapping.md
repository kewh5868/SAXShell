# Electron Density Mapping

The **Electron Density Mapping** tool is SAXSShell's supporting application for
building radial electron-density profiles from XYZ or PDB inputs and, when
needed, turning those profiles into q-space scattering estimates. The current
UI supports three working styles:

- Single-structure inspection from one XYZ or PDB file.
- Folder averaging across many structures in one directory.
- Cluster-folder workflows for Born-approximation component building, where each
  stoichiometry gets its own density, solvent, Fourier, and saved-output state.

In Project Setup, this is the linked component-build workspace for
**Born Approximation (Average)**.

In the main SAXS workflow the tool can run either in **Preview Mode** or in
**Computed Distribution Mode**. Preview mode is for exploratory work. Computed
distribution mode is the linked workflow launched from **Build SAXS Components**
and is the only mode that can push Born-approximation components back into the
active distribution.

## Launching the application

### From the terminal

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs.electron_density_mapping
```

### From the main SAXS UI

- **Tools > SAXS Calculation Preview > Open Electron Density Mapping** opens
  the tool in **Preview Mode**. The window title shows
  `Electron Density Mapping (Preview)`, and the banner explains that pushed
  model components are disabled in this mode.
- **Build SAXS Components** with the
  **Born Approximation (Average)** build mode opens the tool in
  **Computed Distribution Mode**. The tool inherits the active project q-range,
  the active computed distribution, the preferred input folder, and the
  distribution output directory.

## Operating modes

| Mode                 | What is loaded                                     | What the tool does                                                                                                                                                                                                 |
| -------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Single file**      | One `.xyz` or `.pdb` file                          | Computes one radial electron-density profile, optional solvent subtraction, and optional Born-approximation Fourier transform.                                                                                     |
| **Structure folder** | One folder of `.xyz` / `.pdb` files                | Averages the profiles across the folder, tracks variance, and can optionally use contiguous-frame grouping when filenames follow `frame_<NNNN>`.                                                                   |
| **Cluster folder**   | One folder whose subfolders are stoichiometry bins | Builds one state row per stoichiometry, lets you inspect each row independently, and supports batch or manual per-row workflows. Single-atom rows skip density generation and use direct Debye scattering instead. |

## What the tool computes

For density-based rows, the workflow:

1. Loads atom positions and element symbols from XYZ or PDB files.
2. Computes the mass-weighted center of mass using atomic data from `xraydb`.
3. Re-centers each structure using the active center mode.
4. Partitions the structure into a spherical mesh of radial shells and angular
   cells.
5. Accumulates each atom's atomic number as its electron count on that mesh.
6. Averages angular contributions into a radial profile `rho(r)`.
7. Optionally applies Gaussian smearing.
8. Optionally applies a flat solvent electron-density contrast and finds the
   highest-r cutoff crossing.
9. Optionally evaluates a spherical Born-approximation transform into `I(q)`.

!!! note "Raw density versus shell bookkeeping"
The plotted `rho(r)` and the derived Fourier/scattering outputs use a
finite-radius shell-overlap density profile. Exported
`electrons_in_shell` / `shell_electron_counts` values remain point-tagged
shell bookkeeping totals. This distinction matters when an atom sits near
the active origin: the finite-radius profile avoids an artificial
near-origin spike that would otherwise exaggerate the perceived electron
density and distort the scattering estimate.

For single-atom stoichiometry rows in cluster-folder mode, the density step is
skipped and the tool evaluates a direct Debye scattering profile instead.

When folder averaging uses **Use Contiguous Frame Evaluation**, the tool groups
filenames by `frame_<NNNN>` sequences when available and locks each contiguous
set to a shared center offset relative to the heaviest-element anchor. If the
expected naming pattern is not present, the run falls back to complete
averaging and records that fallback in the status log.

## Left-panel controls

### Input

| Field                               | Description                                                                                               |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Path field**                      | Path to a single XYZ/PDB file, a folder of structures, or a folder of stoichiometry subfolders.           |
| **Choose File** / **Choose Folder** | Open file-system pickers for the active input.                                                            |
| **Load Input**                      | Inspects the selected path and populates the window state.                                                |
| **Input mode**                      | Shows whether the current input is a file, a folder, or cluster folders with an active stoichiometry row. |
| **Reference file**                  | The structure file shown in the viewer for the currently active row.                                      |
| **Structure summary**               | Reports atom counts, element counts, center information, and domain size for the active structure.        |

!!! note "Cluster folders"
When a cluster-folder input is loaded, the stoichiometry table becomes the
main navigation surface for the tool. Clicking a row updates the plots,
mesh controls, Fourier controls, and viewer to that row's state.

### Output

| Field                | Description                                                                                                                                                                                                                                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Output directory** | Where run outputs are written when file export is enabled. In preview mode the default is inferred from the loaded input, or from `<project>/electron_density_mapping` when a project-linked launch is available. In computed-distribution mode, the default is `<distribution>/electron_density_mapping`. |
| **Output basename**  | Stem for exported density files. File and folder runs write `<basename>.csv` and `<basename>.json`. Cluster-folder runs append a stoichiometry suffix, for example `<basename>_PbI2.csv`.                                                                                                                  |

### Mesh Settings

The mesh controls define the radial sampling domain before density accumulation.

| Field                                                            | Description                                                                                               |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **rstep (A)**                                                    | Width of the radial shells. Smaller values increase radial resolution.                                    |
| **Theta divisions** / **Phi divisions**                          | Angular subdivision of the spherical mesh.                                                                |
| **rmax (A)**                                                     | Maximum radial extent of the mesh. Atoms outside this radius are excluded and reported in the status log. |
| **Center mode**                                                  | Read-only summary of the active centering choice.                                                         |
| **Calculated center**                                            | Mass-weighted center of mass.                                                                             |
| **Active center**                                                | The origin currently used by the calculation and viewer.                                                  |
| **Nearest atom**                                                 | The atom nearest to the calculated center of mass.                                                        |
| **Reference element**                                            | Element used for the reference-element geometric center when that center mode is selected.                |
| **Total-atom geometric center**                                  | Geometric center of all atoms in the active reference structure.                                          |
| **Reference-element geometric center**                           | Geometric center of only the chosen reference element.                                                    |
| **Reference-center offset**                                      | Offset between the total-atom and reference-element geometric centers.                                    |
| **Calculated Center** / **Nearest Atom** / **Reference Element** | The three center-snap buttons. Exactly one is active at a time.                                           |
| **Update Mesh Settings**                                         | Applies the pending mesh values to the active row.                                                        |
| **Active mesh**                                                  | Summary of the currently applied mesh.                                                                    |
| **Pending fields**                                               | Highlights which mesh fields differ from the applied mesh.                                                |

!!! warning "Mesh locking in manual cluster runs"
In cluster-folder **Manual Mode**, once a manual density calculation has
succeeded, mesh settings remain locked until **Reset Calculated Densities**
is used.

### Actions

The **Actions** group now covers both simple runs and cluster-folder workflows.

| Control                                       | Description                                                                                                                                                                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Stoichiometry Table**                       | Appears for cluster-folder inputs. Columns report stoichiometry, file count, average atoms, density status, Fourier status, trace color, smearing, solvent density, solvent cutoff, Fourier settings summary, and reference file. |
| **Use Contiguous Frame Evaluation**           | Enables contiguous-frame grouping for multi-structure folder and cluster-folder runs.                                                                                                                                             |
| **Manual Mode (selected stoichiometry only)** | Runs only the selected row instead of the whole cluster table.                                                                                                                                                                    |
| **Computed stoichiometries indicator**        | Shows overall completion state for cluster-folder work.                                                                                                                                                                           |
| **Run Electron Density Calculation**          | Runs the active file/folder input, or the active row / entire table in cluster mode depending on Manual Mode.                                                                                                                     |
| **Stop Active Calculation**                   | Cancels the active background calculation.                                                                                                                                                                                        |
| **Reset Calculated Densities**                | Clears density, solvent, Fourier, saved-output, and session-restore state after secondary confirmation.                                                                                                                           |
| **Overall progress**                          | Visible during cluster-folder batch runs and reports completed stoichiometry groups.                                                                                                                                              |
| **Progress bar / message**                    | Shows detailed stage progress for the active run or workspace load.                                                                                                                                                               |

### Smearing

The **Smearing** section applies a Gaussian kernel to the active density
profile. The Debye-Waller factor is live and does not require rerunning the
density calculation.

| Field                                             | Description                                                                                               |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Debye-Waller factor (A^2)**                     | Smearing strength. `0` disables smearing.                                                                 |
| **Gaussian sigma**                                | Read-only sigma derived from the Debye-Waller factor.                                                     |
| **Behavior**                                      | Read-only summary of the active smearing state.                                                           |
| **Apply Smearing**                                | Applies the current smearing to the active row.                                                           |
| **Apply to All**                                  | In cluster-folder mode, applies the current smearing setting to every selected or all stoichiometry rows. |
| **Auto-save smearing snapshots to Saved Outputs** | When enabled, each smearing re-evaluation is captured as a reloadable Saved Outputs entry.                |

### Electron Density Contrast

The **Electron Density Contrast** section estimates a flat solvent density and
subtracts it from the smeared profile.

| Field                                    | Description                                                                                                          |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Compute option**                       | Choose between solvent formula plus density, reference solvent structure, or a direct electron-density value.        |
| **Saved solvents**                       | Preset selector plus save/delete controls for custom solvent presets.                                                |
| **Solvent formula** / **Density (g/mL)** | Used for the solvent-formula method.                                                                                 |
| **Direct density (e-/ A^3)**             | Used for the direct-value method.                                                                                    |
| **Reference solvent file**               | Used for the reference-structure method.                                                                             |
| **Apply Electron Density Contrast**      | Applies the active contrast settings to the current profile.                                                         |
| **Apply to All**                         | Reuses the same solvent settings across the target stoichiometry rows while preserving each row's own cutoff result. |
| **Active contrast**                      | Read-only summary of the active solvent subtraction.                                                                 |
| **Notes**                                | Read-only status for cutoff detection and residual behavior.                                                         |

### Fourier Transform

The **Fourier Transform** section converts the active density profile into a
Born-approximation scattering estimate. The preview panel always shows the
resampled and windowed real-space data that will be used.

The default transform domain is **mirrored mode**, which reflects the profile
about `r = 0` and evaluates the windowed transform over `-rmax` to `rmax`.
The UI also keeps a **legacy r min to r max transform** toggle for historical
behavior.

The evaluated transform is:

$$
F(q) =
4\pi
\int_{r_\min}^{r_\max}
\rho(r)\, W(r)\, r^2\,
\operatorname{sinc}\!\left(\frac{q r}{\pi}\right)
\,\mathrm{d}r
$$

with `I(q) = |F(q)|^2`.

| Field                                            | Description                                                                                                                                                                                |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **r min / r max**                                | Real-space bounds used for the transform.                                                                                                                                                  |
| **Legacy r min to r max transform**              | Switches from the default mirrored-domain transform back to the historical one-sided `r min` to `r max` behavior. In mirrored mode, the left bound is shown as `-r max`.                   |
| **q min / q max / q step**                       | Requested q grid for the output scattering profile.                                                                                                                                        |
| **Window**                                       | Real-space apodization window. Options are `None`, `Lorch`, `Cosine`, `Hanning`, `Parzen`, `Welch`, `Gaussian`, `Sine`, and `Kaiser-Bessel`.                                               |
| **Resample pts**                                 | Number of resampled real-space points used by the transform.                                                                                                                               |
| **Use solvent-subtracted profile**               | Uses the solvent-subtracted smeared density when available.                                                                                                                                |
| **Log q axis** / **Log intensity axis**          | Plot scaling preferences for the scattering plot.                                                                                                                                          |
| **Evaluate Fourier Transform**                   | Evaluates the current transform for the active row, or the target batch in cluster mode.                                                                                                   |
| **Apply to All**                                 | In cluster-folder mode, switches the Fourier settings table into the shared-edit workflow. q settings stay shared while each stoichiometry keeps its own r range and source profile state. |
| **Fourier settings table**                       | Per-stoichiometry summary and edit surface for Fourier settings in cluster mode.                                                                                                           |
| **Available r range** / **Sampling** / **Notes** | Read-only diagnostics for transform support, Nyquist limits, oversampling, clamping, and solvent-profile fallback notes.                                                                   |

!!! note "Single-atom cluster rows"
Single-atom stoichiometry rows do not build an electron-density profile.
Their Fourier action evaluates a direct Debye scattering trace instead.

!!! note "Inherited q-range versus pushed q-range"
In computed-distribution mode, the tool inherits the project's q-range.
If you intentionally evaluate a different q-range here, SAXSShell warns
before push, but the pushed model components still use the transform grid
exactly as written.

### Push to Model

The **Push to Model** group sits directly below **Fourier Transform** and above
**Saved Outputs**. It is available only in computed-distribution mode.

| Control           | Description                                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Status text**   | Explains why push is enabled or disabled.                                                                                       |
| **Push to Model** | Writes the current Born-approximation component traces into the linked computed distribution so the main SAXS UI can load them. |

Push remains disabled when:

- The tool is in preview mode.
- The window is not linked to a computed distribution.
- Any cluster row is still missing its Fourier transform.
- The linked distribution already has saved prefit snapshots.
- The linked distribution already has saved DREAM runs.

The push step is independent from calculation and transform persistence. In
computed-distribution mode, reopening the tool restores saved densities and
Fourier transforms even if **Push to Model** has never been used.

### Debye Scattering Calculation

The **Debye Scattering Calculation** group is a validation workspace for
comparing the current Born-approximation traces against averaged direct Debye
scattering traces on the same q-grid.

| Control                        | Description                                                                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| **Status text**                | Summarizes which active row or target rows already have Debye averages that match the current Born q-grid.                        |
| **Calculate Debye Scattering** | Computes an averaged direct Debye scattering trace on the exact q-grid of the current Born transform.                             |
| **Apply to All Rows**          | In cluster-folder mode, computes Debye averages for every eligible stoichiometry row instead of only the current selection.       |
| **Open Comparison Plot**       | Opens a separate overlay dialog. Born traces use the left axis and solid lines; Debye traces use the right axis and dashed lines. |

Rows are eligible only when they already have a Born-approximation Fourier
trace. Single-atom rows are skipped here because they already use direct Debye
scattering as their main scattering result.

### Saved Output Sets

The **Saved Output Sets** section captures intermediate and final states for
reload and comparison.

| Control                     | Description                                                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Saved Output Sets table** | Records density, solvent-subtraction, and Fourier entries with context, averaging mode, mesh, smearing, solvent, and Fourier summaries. |
| **Load Selected**           | Restores one saved entry back into the main UI state.                                                                                   |
| **Compare Selected**        | Opens a comparison dialog for one or more saved entries with PNG and CSV export tools.                                                  |

Saved Outputs track density, solvent-subtraction, and Fourier snapshots. In
computed-distribution mode, Debye comparison traces are restored from the
workspace session state instead of from the Saved Outputs history.

Persistence paths:

- Preview-mode history is stored in the chosen output directory as
  `electron_density_saved_output_history.json`.
- Computed-distribution history is stored in the linked distribution as
  `electron_density_mapping/saved_output_history.json`.
- Computed-distribution session restore state is stored separately as
  `electron_density_mapping/workspace_state.json`.

That session state preserves calculated densities, Fourier transforms, and
Debye scattering comparison traces even if they have not been pushed to the
model yet. It is cleared only by
**Reset Calculated Densities**.

### Status

The **Status** panel is the persistent log for:

- Input and workspace loading.
- Contiguous-frame grouping or fallback notes.
- Calculation progress and completion summaries.
- Solvent subtraction and Fourier warnings.
- Export, persistence, and push-to-model messages.

## Right panel and viewer

The right-hand side contains shared plot controls followed by the plot panels
and the structure viewer.

### Global plot controls

- **Show Variance Shading** toggles variance bands on density plots.
- **Auto-expand plots** expands panels automatically when their data updates.
- **Show All Cluster Transforms** overlays every ready cluster transform on the
  active scattering plot.
- **Expand All / Collapse All** toggles every plot section.
- **Export Plot Traces** writes the currently displayed raw density, smeared
  density, Fourier preview, and scattering traces into one CSV export.

### Plot panels

| Panel                                                   | Description                                                                                                     |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Orientation-Averaged Radial Electron Density rho(r)** | Finite-radius shell-averaged raw density profile used as the basis for the displayed real-space interpretation. |
| **Gaussian-Smeared Radial Electron Density rho(r)**     | Smoothed density profile after the active smearing setting.                                                     |
| **Solvent-Subtracted Residual**                         | Residual curve after solvent subtraction, including the active cutoff marker when present.                      |
| **Fourier Transform Preview**                           | Resampled and windowed density source that will be used for the scattering transform.                           |
| **Scattering Profile I(q)**                             | Born-approximation or direct Debye scattering result evaluated from the corrected density source.               |

### Structure Viewer

The structure viewer is an interactive **3D** Matplotlib viewer, not a static
2D preview. It supports:

- Rotate, pan, and zoom interactions.
- Reset View.
- Mesh overlay toggling.
- Mesh contrast, line width, and color controls.
- Point-atom display mode.
- Active-origin and zoom readouts over the scene.
- Preserved zoom and camera state while switching between stoichiometry rows in
  cluster-folder mode.

## Typical workflows

### Standalone preview

1. Launch **Open Electron Density Mapping** from the main UI or start the tool
   from the terminal.
2. Load a single structure file or a folder of structures.
3. Review the structure summary and set the mesh.
4. Run the density calculation.
5. Adjust smearing, optional solvent subtraction, and optional Fourier settings.
6. Optionally compute a matching Debye scattering average and open the
   comparison plot.
7. Use **Saved Output Sets** to reload or compare intermediate states.

### Computed distribution / Born approximation

1. Open the tool from **Build SAXS Components** with the
   **Born Approximation (Average)** workflow.
2. Confirm that the inherited cluster folder, output directory, and q-range are
   correct.
3. Review the stoichiometry table and choose whether to run the full batch or
   **Manual Mode**.
4. Run the density calculation.
5. Apply smearing, solvent subtraction, and Fourier settings row-by-row or with
   **Apply to All**.
6. Evaluate Fourier transforms for every required row.
7. Optionally compute Debye scattering averages for the active or target rows
   and inspect the Born-vs-Debye comparison plot.
8. Use **Push to Model** only after all desired transforms are complete.

## Output and persistence files

| File                                                                      | When it appears                                                     | Contents                                                                                                                           |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `<basename>.csv` / `<basename>.json`                                      | File and folder runs with an output directory                       | Density profile export and JSON summary for the run.                                                                               |
| `<basename>_<stoichiometry>.csv` / `<basename>_<stoichiometry>.json`      | Cluster-folder runs with an output directory                        | One density export pair per density-based stoichiometry row. Direct-Debye-only rows do not write this pair during the density run. |
| `electron_density_saved_output_history.json`                              | Preview mode with a writable output directory                       | Saved-output history for reload and comparison.                                                                                    |
| `saved_output_history.json`                                               | Computed-distribution mode                                          | Distribution-linked saved-output history.                                                                                          |
| `workspace_state.json`                                                    | Computed-distribution mode after a density or Fourier result exists | Restorable session state for non-pushed calculations.                                                                              |
| `born_approximation_component_summary.json`                               | After **Push to Model**                                             | Saved pushed component summary for the linked distribution.                                                                        |
| `md_saxs_map.json` or `md_saxs_map_predicted_structures.json`             | After **Push to Model**                                             | SAXS component registry for the linked distribution.                                                                               |
| `scattering_components/` or `scattering_components_predicted_structures/` | After **Push to Model**                                             | Per-component Born-approximation scattering traces written as `.txt` files.                                                        |

## Related pages

- [SAXS Contrast Mode](saxs-contrast-mode.md)
- [Cluster Extraction](cluster-extraction.md)
- [Results and Export](results-and-export.md)
- [XYZ to PDB Conversion](xyz2pdb-conversion.md)

## References

- [Larch XAFS Fourier-transform documentation: practical window definitions including Hanning, Parzen, Welch, Gaussian, Sine, and Kaiser-Bessel.](https://xraypy.github.io/xraylarch/xafs_fourier.html)
- [ORNL X-ray and neutron reflectivity tutorial: kinematic "master formula" discussion linking reciprocal-space scattering to Fourier transforms of real-space interfacial structure.](https://neutrons.ornl.gov/sites/default/files/beamline_04A_xray_neutron_reflectivity_tutorial.pdf)
- [Steinruck H.-G., Han H.-L., et al. _Fluoroethylene Carbonate Induces Ordered Electrolyte Interface on Silicon and Sapphire Surfaces as Revealed by Sum Frequency Generation Vibrational Spectroscopy and X-ray Reflectivity_. Nano Letters (2018).](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.8b00298)
- [Elam W. T., Ravel B. D., Sieber J. R. _A new atomic database for X-ray spectroscopic calculations_. Radiation Physics and Chemistry (2002). Background reference for the `xraydb` atomic quantities used elsewhere in this tool.](<https://doi.org/10.1016/S0969-806X(01)00327-4>)
