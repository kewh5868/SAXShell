# Electron Density Mapping

The **Electron Density Mapping** tool is a supporting application that computes
radial electron-density profiles from XYZ or PDB molecular structures. It can
be launched directly from the `saxshell` main UI or from the terminal.

## Launching the application

### From the terminal

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs.electron_density_mapping
```

### From the SAXSShell main UI

Open the main SAXSShell application and select **Electron Density Mapping** from the supporting applications menu. When launched this way the tool inherits the active project directory and will default its output path to `exported_results/data/` within that project.

---

The tool takes one or more structure files, centers each structure at a chosen
origin, bins the atomic electron counts onto a spherical mesh, and returns an
orientation-averaged radial profile ρ(r). An optional Gaussian-smearing step
and a Fourier-transform stage can convert the real-space profile into a
q-space scattering estimate.

---

## What the tool computes

For each structure, the workflow:

1. Loads atom positions and element symbols from an XYZ or PDB file.
2. Computes the mass-weighted center of mass using atomic masses from `xraydb`.
3. Centers all atom positions relative to the chosen active origin.
4. Partitions the structure domain into a spherical mesh of radial shells
   and angular cells.
5. Accumulates the atomic number of each atom (used as its electron count) into
   the mesh cell it falls into.
6. Averages the electron density over all angular directions within each radial
   shell, yielding the orientation-averaged profile ρ(r).
7. Optionally applies a Gaussian-smearing kernel to ρ(r).
8. Optionally evaluates a spherical Born-approximation Fourier transform of the
   smeared profile to produce a scattering amplitude and intensity I(q).

When a **folder** is provided instead of a single file, the workflow repeats
steps 2–6 for every valid structure in the folder and averages the resulting
profiles across the ensemble, also tracking inter-member variance.

---

## Panels and input fields

The window is split into a left-hand control panel and a right-hand plot area.

### Input

| Field                 | Description                                                                                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Path field**        | Type or paste the path to an XYZ or PDB file, or to a folder of such files. Press Enter or click **Load Input** to apply.                                        |
| **Choose File**       | Opens a file-browser dialog pre-filtered for `.xyz` and `.pdb` files.                                                                                            |
| **Choose Folder**     | Opens a folder-browser dialog. All valid structure files in the folder are discovered automatically.                                                             |
| **Load Input**        | Inspects the selected path, determines the input mode, loads the reference structure, and populates the status fields below.                                     |
| **Input mode**        | Read-only. Shows `file` when a single structure is loaded, or `folder` when a directory is active.                                                               |
| **Reference file**    | Read-only. Shows the file that is used for the 3D structure preview and center-of-mass calculation. In folder mode this is the first file in natural sort order. |
| **Structure summary** | Read-only. Shows atom count, element composition, center of mass, active center, and the domain radius rmax after a structure is loaded.                         |

!!! note "Folder mode"
When a folder is loaded, the **structure viewer** and **center** displays
reflect only the reference (first) file. The **Run** step averages over
every file in the folder and produces ensemble variance bands in the plots.

---

### Output

| Field                | Description                                                                                                                                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Output directory** | Folder where the CSV profile and JSON summary are written after a successful calculation run. Leave blank to skip file output and inspect results interactively only. Click **Browse** to choose a folder. |
| **Output basename**  | Filename stem used for all output files. Defaults to `electron_density_profile`. The tool appends `_profile.csv` and `_summary.json` automatically.                                                        |

---

### Mesh Settings

The spherical mesh controls how the structure domain is discretised before
electron counts are binned.

| Field                           | Description                                                                                                                                                                                                                                                                                              |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **rstep (Å)**                   | Width of each radial shell in ångströms. Smaller values give a finer radial resolution at the cost of a larger mesh. Default: 0.1 Å.                                                                                                                                                                     |
| **Theta divisions**             | Number of equal-angle bins along the polar (θ) axis from 0 to π. Higher values improve angular coverage. Default: 120.                                                                                                                                                                                   |
| **Phi divisions**               | Number of equal-angle bins along the azimuthal (φ) axis from 0 to 2π. Default: 60.                                                                                                                                                                                                                       |
| **rmax (Å)**                    | Maximum radial extent of the mesh in ångströms. Atoms beyond this radius are excluded from the profile and counted in the excluded-atom summary. Set this to at least the structural domain radius reported in the structure summary. Default: 8.0 Å (updated automatically when a structure is loaded). |
| **Center mode**                 | Read-only label showing whether the active origin is the `Calculated center of mass` or the `Nearest atom` to the center of mass.                                                                                                                                                                        |
| **Calculated center**           | Read-only. The mass-weighted center of mass of the reference structure in Cartesian coordinates (Å).                                                                                                                                                                                                     |
| **Active center**               | Read-only. The origin actually used for centering. Matches the calculated center unless the center has been snapped to the nearest atom.                                                                                                                                                                 |
| **Nearest atom**                | Read-only. The element symbol, atom index, and distance (Å) of the atom closest to the calculated center of mass.                                                                                                                                                                                        |
| **Snap Center to Nearest Atom** | Moves the active origin from the calculated center of mass to the position of the nearest atom. Useful when the center of mass falls in an empty space between atoms and you prefer a physically grounded lattice site as the origin. The profile and viewer update immediately.                         |
| **Reset to Calculated Center**  | Restores the active origin to the mass-weighted center of mass.                                                                                                                                                                                                                                          |
| **Update Mesh Settings**        | Applies the current rstep, theta, phi, and rmax values to the active mesh and regenerates the profile if a calculation result is already loaded.                                                                                                                                                         |
| **Active mesh**                 | Read-only summary of the mesh that was used for the most recent calculation: rstep, theta divisions, phi divisions, rmax.                                                                                                                                                                                |
| **Pending fields**              | Read-only. If any mesh field has been changed since the last **Update Mesh Settings**, the changed fields are listed here as a reminder that the active mesh and the control values are out of sync.                                                                                                     |

---

### Actions

| Control                              | Description                                                                                                                                                                                                                   |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Run Electron Density Calculation** | Starts a background worker that processes all structure files under the active input path using the current mesh settings, center mode, and smearing settings. The progress bar and message update as each file is processed. |
| **Progress bar / message**           | Shows the number of structures processed and a human-readable stage description during the calculation. Reads "Idle" when no calculation is running.                                                                          |

!!! warning "Mesh rmax and atom exclusion"
If **rmax** is smaller than the actual domain of the structure, atoms that
fall outside the mesh will be excluded from the profile. The number of
excluded atoms and their total electron count are reported in the status log
after the run completes. This is expected if you deliberately want to
restrict the profile to a sub-domain, but will produce an incomplete profile
if rmax is simply left at a default value smaller than the structure.

---

### Smearing

The smearing section applies a Gaussian kernel to the raw radial profile.
This is useful for softening the sharp step transitions between shells when
comparing to experimental SAXS data or preparing a profile for Fourier
transformation.

<!-- prettier-ignore-start -->

| Field | Description |
|---|---|
| **Debye-Waller factor (Å²)** | Controls the width of the Gaussian kernel via $\sigma = \sqrt{B}$, where $B$ is this value. A value of 0 disables smearing entirely. Default: 0.006 Å². |
| **Gaussian sigma** | Read-only. Shows the equivalent Gaussian standard deviation $\sigma$ in ångströms derived from the Debye-Waller factor. |
| **Behavior** | Read-only. Confirms whether smearing is active or disabled and summarises the active σ value. |

The smearing is applied live whenever the Debye-Waller factor is changed. It
does not require re-running the full calculation.

The relation between the Debye-Waller factor $B$ and the Gaussian standard
deviation $\sigma$ is:

$$
\sigma = \sqrt{B}
$$

<!-- prettier-ignore-end -->

---

### Fourier Transform

The Fourier Transform section converts the smeared radial density profile into
a q-space scattering estimate using a spherical Born approximation for a
radially averaged density. In this implementation the transform is evaluated on
the selected real-space interval after the profile is resampled onto a uniform
grid. The preview plot shows exactly what profile enters the numerical
integration.

The transform assumes spherical symmetry at the level of the plotted profile,
so it should be interpreted as a first-pass reciprocal-space estimate rather
than a full anisotropic scattering calculation.

The transform evaluates:

<!-- prettier-ignore-start -->

$$
F(q) =
4\pi
\int_{r_\min}^{r_\max}
\rho(r)\, W(r)\, r^2\,
\operatorname{sinc}\!\left(\frac{q r}{\pi}\right)
\,\mathrm{d}r
$$

where $W(r)$ is the optional real-space window function and
$I(q) = |F(q)|^2$.

The code explicitly supports $r_\min = 0$. This is numerically well behaved
for the spherical kernel above because the sinc factor approaches 1 as
$r \to 0$, while the additional $r^2$ factor keeps the origin contribution
finite.

Window functions are applied over the selected transform interval. The
interval-centered families such as Hanning, Welch, Gaussian, Parzen, Sine, and
Kaiser-Bessel are peaked at the midpoint of the chosen range and taper toward
the interval boundaries. This mirrors the conventional finite-range
apodisation strategy used in EXAFS workflows, while the Lorch option provides
the familiar sinc-like damping often used to reduce truncation ripples.

A **preview plot** shows the resampled and windowed profile before the
integral is evaluated. This is useful for checking that the r-range and window
settings are appropriate before committing to a full transform.

| Field | Description |
|---|---|
| **r min (Å)** | Lower bound of the r-range used for the transform. The portion of the profile below this value is excluded. Default: 0.0 Å. |
| **r max (Å)** | Upper bound of the r-range used for the transform. Should not exceed the mesh rmax or the structural domain. Default: 1.0 Å (auto-updated from the mesh). |
| **Transform window** | Apodisation function applied to the profile before transformation to reduce truncation ringing. Options: `None` (rectangular, no tapering), `Lorch`, `Cosine`, `Hanning`, `Parzen`, `Welch`, `Gaussian`, `Sine`, and `Kaiser-Bessel`. Default: `None`. |
| **Resampling points** | Number of uniformly-spaced r points used when interpolating the profile before the numerical integration. Higher values improve transform accuracy. Default: 1024. |
| **q min (Å⁻¹)** | Minimum q value of the output scattering profile. Default: 0.02 Å⁻¹. |
| **q max (Å⁻¹)** | Maximum q value of the output scattering profile. The tool may clamp this to the Nyquist limit determined by the resampling step. Default: 10.0 Å⁻¹. |
| **q step (Å⁻¹)** | Spacing between q grid points in the output profile. Default: 0.02 Å⁻¹. |
| **Use solvent-subtracted profile** | When checked, the transform uses the solvent-subtracted smeared density if a solvent electron density has already been computed. If no solvent contrast is active, the tool falls back to the ordinary smeared profile and notes that fallback in the preview. |
| **Log q axis** | When checked, the q axis of the scattering plot uses a logarithmic scale. |
| **Log intensity axis** | When checked, the intensity axis of the scattering plot uses a logarithmic scale. |
| **Evaluate Fourier Transform** | Evaluates the transform using the current settings and updates the scattering plot. Requires a completed density calculation. |
| **Available r range** | Read-only. Reports the r range available from the current profile result, i.e., the actual mesh domain after extending the transform support to the origin. |
| **Sampling** | Read-only. Reports the resampling step size, the Nyquist q limit derived from that step, and the independent q step size. Warns if the q grid is oversampled relative to the Nyquist limit. |
| **Notes** | Read-only. Flags any conditions that may affect the transform quality, such as a clamped q max, a q grid that is significantly oversampled, or a fallback from solvent-subtracted to ordinary smeared density. |

#### Window guidance

- `None` is the sharp-cutoff choice. It preserves the selected interval exactly but is the most prone to ringing from finite-range truncation.
- `Lorch` is useful when you want a sinc-like damping toward the upper bound and are specifically trying to suppress termination ripples.
- `Hanning`, `Welch`, `Parzen`, `Gaussian`, `Sine`, and `Kaiser-Bessel` are centered interval windows. They are strongest in the middle of the selected range and taper toward both ends.
- `Kaiser-Bessel` is often a good general-purpose compromise when you want stronger sidelobe suppression than a simple cosine-family window.
- If you need low-r features preserved as strongly as possible, keep the transform range physically justified and compare more than one window in the preview before committing to an interpretation.

#### Sampling guidance

- Increasing **Resampling points** decreases the uniform $\Delta r$ used in the transform and therefore raises the Nyquist-limited maximum usable q.
- Decreasing the real-space interval width increases the independent q spacing, so a very fine requested **q step** may oversample the plotted curve without adding independent information.
- Starting the transform at $r = 0$ is supported, but if the low-r bins are dominated by a center-snapped atom or by strong solvent subtraction, compare the preview carefully before interpreting the high-q tail.

<!-- prettier-ignore-end -->

---

### Status

A scrollable log box that records loaded structures, settings applied, progress
messages, and any warnings about excluded atoms or transform quality. Useful for
reviewing what settings were active during a calculation.

---

## Right-panel plots

The right side of the window contains six stacked panels.

| Plot                                                  | Description                                                                                                                                                                                    |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Orientation-Averaged Radial Electron Density ρ(r)** | Raw binned profile before smearing. Displayed as a step trace with an optional variance shading band (one standard deviation across ensemble members).                                         |
| **Gaussian-Smeared Radial Electron Density ρ(r)**     | The same profile after the Gaussian-smearing kernel has been applied. Displayed as a smooth curve with optional variance shading.                                                              |
| **Solvent-Subtracted Residual**                       | The smeared profile minus the active flat solvent electron density, shown after solvent contrast has been computed. The highest-r crossing used for the solvent cutoff is marked here as well. |
| **Fourier Transform Preview**                         | Shows the resampled and windowed density data that will be fed into the transform integral. Updates live when the Fourier Transform settings change.                                           |
| **Scattering Profile I(q)**                           | The resulting Born-approximation intensity after **Evaluate Fourier Transform** is clicked.                                                                                                    |
| **Structure Viewer**                                  | A 2D projected scatter plot of atom positions from the reference structure, coloured by element. The active center of mass origin is shown as a reference point.                               |

The **Show Variance Shading** checkbox above the plots toggles the standard-deviation
band on both the raw and smeared profile plots simultaneously.

---

## Typical workflow

1. Open the tool from the main SAXSShell UI or launch `saxshell` with the
   `electron-density-mapping` subcommand.
2. In **Input**, choose a single XYZ or PDB file or a folder of structures.
   Click **Load Input**.
3. Review the **Structure summary** field: check the element composition,
   center of mass, and domain rmax.
4. In **Mesh Settings**, set **rmax** to at least the reported domain radius.
   Adjust **rstep**, **Theta divisions**, and **Phi divisions** as needed for
   the desired resolution.
5. Choose a **Center mode**. If the center of mass falls in an interstitial
   void, use **Snap Center to Nearest Atom** to anchor the origin to a real
   atomic site.
6. Click **Update Mesh Settings** to apply any field changes.
7. Set the **Output directory** and **Output basename** if you want to save
   results to disk.
8. Click **Run Electron Density Calculation** and wait for the progress bar to
   complete.
9. Inspect the raw and smeared ρ(r) plots. Adjust the **Debye-Waller factor**
   under **Smearing** to control profile smoothness.
10. If a q-space estimate is needed, configure the **Fourier Transform** section
    and click **Evaluate Fourier Transform**.

---

## Output files

When an output directory is specified, two files are written after a successful
run.

| File                      | Contents                                                                                                                                                                                                             |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<basename>_profile.csv`  | Radial centers, orientation-averaged density, smeared density, and per-shell electron counts and volumes.                                                                                                            |
| `<basename>_summary.json` | Metadata including the input path, input mode, element counts, center of mass, active center, center mode, mesh settings, smearing settings, excluded atom/electron counts, and per-member summaries in folder mode. |

---

## Related pages

- [SAXS Contrast Mode](saxs-contrast-mode.md)
- [XYZ to PDB Conversion](xyz2pdb-conversion.md)
- [Cluster Extraction](cluster-extraction.md)
- [Results and Export](results-and-export.md)

## References

- [Larch XAFS Fourier-transform documentation: practical window definitions including Hanning, Parzen, Welch, Gaussian, Sine, and Kaiser-Bessel.](https://xraypy.github.io/xraylarch/xafs_fourier.html)
- [ORNL X-ray and neutron reflectivity tutorial: kinematic “master formula” discussion linking reciprocal-space scattering to Fourier transforms of real-space interfacial structure.](https://neutrons.ornl.gov/sites/default/files/beamline_04A_xray_neutron_reflectivity_tutorial.pdf)
- [Steinrück H.-G., Han H.-L., et al. _Fluoroethylene Carbonate Induces Ordered Electrolyte Interface on Silicon and Sapphire Surfaces as Revealed by Sum Frequency Generation Vibrational Spectroscopy and X-ray Reflectivity_. Nano Letters (2018).](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.8b00298)
- [Elam W. T., Ravel B. D., Sieber J. R. _A new atomic database for X-ray spectroscopic calculations_. Radiation Physics and Chemistry (2002). Background reference for the `xraydb` atomic quantities used elsewhere in this tool.](<https://doi.org/10.1016/S0969-806X(01)00327-4>)
