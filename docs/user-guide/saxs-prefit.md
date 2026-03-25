# SAXS Prefit

The **SAXS Prefit** tab is where the project turns into a concrete model
preview. It combines:

- the selected template
- the current parameter table
- the experimental dataset
- built SAXS components
- optional cluster geometry metadata

## What you can do in Prefit

From the current UI implementation, Prefit supports:

- updating the model without leaving the tab
- autoscaling the plot
- toggling experimental, model, and solvent traces
- viewing fit metrics directly on the plot
- exporting plot data
- saving and restoring Prefit state
- saving a **Best Prefit** parameter preset
- using a solution-based solute volume-fraction estimator when the active
  template exposes a solute or solvent fraction parameter

## Cluster geometry metadata

Some templates need cluster geometry metadata before they can run.

When enabled, the cluster geometry section lets you:

- compute geometry metadata from the current cluster folder
- map geometry rows onto component weight parameters
- choose ionic or bond-length radius modes
- switch between supported shape approximations
- edit active radii manually
- update the model after edits

Recent updates in the codebase also include:

- progress feedback during geometry computation
- collapsed notes and paths in the geometry table
- positive-radius validation with explicit error reporting
- template-aware restrictions on which shape approximations are allowed

## Solute volume-fraction estimator

If the active template includes a solute or solvent fraction parameter such as
`phi_solute`, Prefit can show an embedded **Solute Volume Fraction Estimator**
between the main controls and the cluster-geometry table.

The current implementation uses the measured solution volume together with the
input solute density:

- `c_solute = m_solute / V_solution`
- `vbar_solute ~= 1 / rho_solute`
- `phi_solute ~= c_solute * vbar_solute`

This is closer to the concentration-plus-specific-volume logic commonly used in
solution SAXS than the older additive-volume estimate
`V_solute / (V_solute + V_solvent)`.

The tool still reports additive component volumes as a diagnostic, but the
value written back into the Prefit parameter table is now the measured-solution
estimate above.

## Plot controls

The Prefit plot now mirrors the DREAM plot more closely than older versions.
You can toggle:

- experimental data
- model curve
- solvent contribution

The plot also shows fit metrics in the lower-left corner.

## Why many templates include `scale` and `offset`

Many SAXS templates in this repository include both a multiplicative `scale`
term and an additive `offset` term.

### `scale`

It is reasonable to ask why a fitted scale factor is still useful even when
experimental data have been reduced onto an absolute intensity scale.

The short answer is:

- absolute scaling is still valuable and should be used when available
- but the model amplitude can still be uncertain even on an absolute scale

For simple absolute-scale sphere models, the SAXS intensity contains the
contrast term explicitly, and the fitted scale should correspond to the volume
fraction only if the model and contrast are both correct. SasView documents
this directly for the sphere model, where the intensity contains the
scatterer-solvent contrast term `Δρ` and notes that, on absolute-scale data,
`scale` should match the volume fraction only when the fit is physically
consistent; otherwise an extra rescaling factor is still needed.

That same practical issue becomes even more important when SAXShell is fitting
simulated or cluster-derived component profiles. In those workflows, the
effective contrast between the cluster, the surrounding medium, and any
solvation or hydration layer is often not known exactly. Published SAXS work on
calculated profiles shows that small changes in hydration-layer contrast can
change the predicted SAXS curve substantially, and implicit-solvent SAXS
calculations often need free parameters to represent excess solvation-layer
density.

So the scientifically careful statement is:

- **do not ignore absolute scaling**
- but **do not assume absolute scaling removes the need for a fitted scale
  factor**

In SAXShell, `scale` is therefore kept in most templates as a practical way to
absorb uncertainty in:

- effective contrast between solute and medium
- hydration or solvation-layer density
- concentration or volume-fraction mismatch
- transmission, thickness, or normalization mismatch

### `offset`

`offset` is used as a flat, q-independent baseline term. This is a practical
background-correction parameter, not a structural parameter.

It is useful when the measured curve still contains an approximately constant
residual background after reduction, for example from:

- imperfect background subtraction
- residual instrument or detector background
- parasitic scattering from sample-environment hardware such as windows,
  holders, or other beamline components

Parasitic scattering from the sample environment is a real and documented issue
in SAXS instrumentation. For example, Edwards-Gayle et al. discuss how sample
holder geometry and materials can introduce measurable parasitic background and
how redesigning the sample cell reduced that contribution.

Fluorescence can also contribute to residual X-ray background in some
experiments, depending on beam energy and sample composition, but the safest
general statement for SAXS fitting is that `offset` absorbs an approximately
flat residual background regardless of its exact origin.

## Practical advice on `scale` and `offset`

- If your data are truly on a reliable absolute scale and the model is fully
  physical, the fitted `scale` should be easier to interpret.
- If `scale` drifts far from expectation, check contrast assumptions,
  normalization, solvent subtraction, and template parameterization before
  over-interpreting the fitted value.
- Use `offset` for flat residual background.
- If the residuals are clearly sloped or curved in q, a constant `offset`
  alone is probably not the right correction.

## Geometry-aware templates

Templates such as the poly-LMA workflows can require geometry metadata. In that
case Prefit will deliberately block model updates until the required metadata
exists and is mapped correctly.

## Practical advice

- Compute geometry metadata once for a project, then recompute only when the
  cluster folder changes or when you want refreshed defaults.
- Treat manual geometry edits as part of the current Prefit state.
- If a geometry-aware template refuses to update, check the mapping column and
  the active radii values before looking elsewhere.

## References

- [SasView sphere model: absolute-scale interpretation of `scale`, explicit contrast term, and flat `background`.](https://www.sasview.org/docs/user/models/sphere.html)
- [SasView power-law model: example of a flat additive background term.](https://www.sasview.org/docs/user/models/power_law.html)
- [Schneidman-Duhovny D, Hammel M, Tainer JA, Sali A. _Accurate SAXS profile computation and its assessment by contrast variation experiments_. Biophysical Journal (2013).](https://pubmed.ncbi.nlm.nih.gov/23972848/)
- [Henriques J, Arleth L, Lindorff-Larsen K, Skepö M. _On the Calculation of SAXS Profiles of Folded and Intrinsically Disordered Proteins from Computer Simulations_. Journal of Molecular Biology (2018).](https://pubmed.ncbi.nlm.nih.gov/29548755/)
- [Edwards-Gayle CJC, Khunti N, Hamley IW, Inoue K, Cowieson N, Rambo RP. _Design of a multipurpose sample cell holder for the Diamond Light Source high-throughput SAXS beamline B21_. Journal of Synchrotron Radiation (2021).](https://pmc.ncbi.nlm.nih.gov/articles/PMC7842227/)
- [Hajizadeh N, Franke D, Jeffries CM, Svergun DI. _Consensus Bayesian assessment of protein molecular mass from solution X-ray scattering data_. Scientific Reports (2018).](https://www.nature.com/articles/s41598-018-25355-2)
