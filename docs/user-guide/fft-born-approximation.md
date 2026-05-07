# 3D FFT Born Approximation

The **3D FFT Born Approximation** tool is SAXSShell's separate Cartesian
Fourier workflow for building scattering curves directly from a voxelized
three-dimensional electron-density map.

Use this tool when you want to keep the full 3D structure instead of reducing
it to a spherically averaged radial density first. That makes it the right
place to experiment with constant solvent-density contrast subtraction in a way
that stays tied to the actual 3D molecular shape.

## Launching the application

### From the main SAXS UI

- **Tools > SAXS Calculation Preview > Open 3D FFT Born Approximation** opens
  the tool in **Preview Mode**.
- **Build SAXS Components** with the
  **3D FFT Born Approximation** build mode opens the tool in
  **Computed Distribution Mode**.

Both launch paths inherit the active project q-range when that information is
available from the main UI.

When the tool is opened from **Build SAXS Components**, it also inherits the
active project structure-source preference:

- **Average cluster folders / input structures** builds each profile from every
  structure file in the matching cluster folder.
- **Representative structures** builds each profile from the single saved
  representative file recorded in
  `rmcsetup/representative_structures/representative_selection.json`.

If no-solvent, partial/source, or full-solvent representative variants are
available, the **Representative solvent** selector lets you choose which saved
variant is used for the 3D FFT run.

## Layout

The window follows the same broad layout style as the 1D Born tool:

- a scrollable **left pane** for input, FFT settings, electron-density
  contrast setup, overlay choices, plot options, actions, and the status log
- a scrollable **right pane** for the structure viewer, q-space curves, the
  FFT real-space visualizer, shell diagnostics, and the run summary

!!! info "Image placeholder"
Add a screenshot of the 3D FFT Born Approximation window showing the split
left and right panes, the q-space plot, and the FFT real-space visualizer.

## 1D versus 3D Born

The two Born tools do related but different jobs.

| Workflow                            | What is transformed                                         | What is preserved                          | Typical use                                                                                                 |
| ----------------------------------- | ----------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **1D Born Approximation (Average)** | A spherically averaged radial density profile `rho(r)`      | Radial structure only                      | Fast legacy workflow, radial diagnostics, comparison to the historical SAXSShell behavior                   |
| **3D FFT Born Approximation**       | A full Cartesian contrast-density grid `Delta rho(x, y, z)` | Full 3D structure before q-shell averaging | 3D density studies, constant solvent-density subtraction, comparison against exact Debye and legacy 1D Born |

In plain language:

- the **1D Born** workflow first averages the structure into a radial profile
  and then Fourier-transforms that profile
- the **3D FFT Born** workflow keeps the full 3D map, Fourier-transforms that
  map, and only then averages intensity over q-shells

## Mathematical model

### Continuous 3D Born amplitude

For a contrast density `Delta rho(r)`, the Born amplitude is

$$
A(\mathbf{q}) =
\int \Delta \rho(\mathbf{r})
\exp\!\left(i \mathbf{q} \cdot \mathbf{r}\right)
\mathrm{d}\mathbf{r}
$$

and the orientationally averaged scattering intensity is

$$
I(q) =
\left\langle
\left|A(\mathbf{q})\right|^2
\right\rangle_{\lVert \mathbf{q} \rVert = q}.
$$

### Constant solvent-density contrast

The current 3D FFT workflow uses the contrast-density form

$$
\Delta \rho(\mathbf{r}) =
\rho_{\mathrm{atom}}(\mathbf{r})
- \rho_0 \chi(\mathbf{r}),
$$

where:

- `rho_atom(r)` is the voxelized atomic electron-density map
- `rho_0` is the constant solvent electron density in `e / Å^3`
- `chi(r)` is the exclusion mask built from the union of atomic exclusion
  spheres

This is why the 3D FFT workflow is the correct place for constant solvent
subtraction: the subtraction is applied in real space on the 3D density field,
not retrofitted into a purely radial post-processing step.

### Discrete FFT form

On a Cartesian grid with voxel volume `Delta V`, the tool evaluates

$$
A(\mathbf{q}_{ijk}) \approx
\Delta V
\sum_n
\Delta \rho(\mathbf{r}_n)
\exp\!\left(i \mathbf{q}_{ijk} \cdot \mathbf{r}_n\right).
$$

The FFT frequencies are converted with

$$
\mathbf{q} = 2 \pi \mathbf{f},
$$

so the q values remain in `Å^-1` when the real-space coordinates are in `Å`.

After the FFT, the tool computes shell-averaged intensity:

$$
I(q_t) =
\left\langle
\left|A(q_x, q_y, q_z)\right|^2
\right\rangle_{q_t}.
$$

That last step is what makes the result comparable to orientationally averaged
Debye scattering.

### Relation to the 1D Born workflow

The 1D Born tool uses a radial transform of the form

$$
F(q) =
4 \pi
\int
\rho(r)\,W(r)\,r^2\,
\mathrm{sinc}\!\left(\frac{q r}{\pi}\right)
\mathrm{d}r
$$

with

$$
I(q) = |F(q)|^2.
$$

That is useful when the radial density itself is the object you want to model.
It is not the same as taking a full 3D FFT of the original molecular density
and then performing q-shell averaging.

## Input fields and current defaults

### 3D FFT Settings

These defaults match the current backend debug and benchmark tests, except that
`q min` and `q max` are inherited from the main UI when available.

| Field                      | Default                     | Meaning                                                            |
| -------------------------- | --------------------------- | ------------------------------------------------------------------ |
| **q min (Å^-1)**           | inherited, otherwise `0.01` | Lower bound of the shared q grid                                   |
| **q max (Å^-1)**           | inherited, otherwise `1.20` | Upper bound of the shared q grid                                   |
| **q step (Å^-1)**          | `0.01`                      | q-grid spacing                                                     |
| **Voxel spacing (Å)**      | `2.5`                       | Cartesian voxel spacing used for the FFT density grid              |
| **Gaussian sigma (Å)**     | `0.75`                      | Atomic deposition width for the voxelized density                  |
| **Minimum box length (Å)** | `640.0`                     | Minimum FFT box length before padding and odd-grid rounding        |
| **Extra padding (Å)**      | `24.0`                      | Additional vacuum padding around the structure before voxelization |

### Electron Density Contrast

The contrast section is separate from the FFT settings so you can configure and
apply solvent subtraction explicitly before the next FFT run.

| Field                            | Default                     | Meaning                                                                                   |
| -------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------- |
| **Compute option**               | solvent formula and density | Chooses the contrast-density setup path                                                   |
| **Saved solvents**               | `Water`                     | Preset solvent formula and density entry                                                  |
| **Solvent formula**              | `H2O` when Water is loaded  | Stoichiometry for neat-solvent estimation                                                 |
| **Density (g/mL)**               | `1.0` when Water is loaded  | Bulk density for neat-solvent estimation                                                  |
| **Direct density (e-/Å^3)**      | `0.334`                     | Manual solvent electron density                                                           |
| **Reference solvent file**       | empty                       | XYZ or PDB file used to estimate a uniform reference density from its full coordinate box |
| **Exclusion radius scale**       | `1.0`                       | Multiplier applied to the atomic exclusion radii                                          |
| **Exclusion radius padding (Å)** | `0.0`                       | Extra radius added to each exclusion sphere                                               |
| **Active contrast**              | none until applied          | The density that will actually be used on the next run                                    |

### Comparison and plot defaults

| Field                                       | Default | Meaning                                                  |
| ------------------------------------------- | ------- | -------------------------------------------------------- |
| **Overlay 1D Born Approximation (Average)** | on      | Computes and displays the legacy radial comparison curve |
| **Overlay exact Debye scattering**          | off     | Computes the exact Debye comparison trace on demand      |
| **Show kernel-corrected FFT overlay**       | off     | Diagnostic overlay for zero-contrast runs only           |
| **Log q axis**                              | on      | Default q-space display scaling                          |
| **Log intensity axis**                      | on      | Default intensity display scaling                        |

## Kernel correction

Kernel correction is a **diagnostic**, not a production solvent-contrast step.

When the 3D density map is built by depositing atoms as Gaussians, that
deposition introduces a known smoothing response in q-space. For a zero-contrast
run, the current backend can divide out that Gaussian intensity factor so the
FFT result is easier to compare with the point-scatterer Debye limit.

For solvent-contrast calculations, leave kernel correction off. Once constant
solvent-density subtraction is active, a single global Gaussian correction is
no longer the physically clean description of the full contrast-density field.

## Outputs

After a run, the 3D FFT window reports:

- the q-space scattering curves
- an optional overlay against the legacy 1D Born and exact Debye curves
- an FFT real-space visualizer showing the centered structure and FFT box
- q-shell population diagnostics
- run timing, Nyquist limit, density integrals, and contrast metadata
- CSV export of the currently displayed q-space curves

In **Computed Distribution Mode**, **Push to Model** writes the computed traces
and component map into the linked computed distribution so the main SAXS UI can
load them for SAXS Prefit and SAXS DREAM Fit. The saved distribution metadata
records whether the traces were built from average folders or representative
structures.

Bare single-atom clusters use a direct single-atom Born trace for the 3D FFT
result so low-q bins do not become empty FFT-shell `NaN` values.

## Related pages

- [1D Born Approximation (Average)](electron-density-mapping.md)
- [GUI Overview](gui-overview.md)
- [Project Setup](../getting-started/project-setup.md)
