# SAXS Prefit

The **SAXS Prefit** tab is where the project turns into a concrete model
preview. It combines:

- the selected template
- the current parameter table
- the experimental dataset
- built SAXS components
- optional cluster geometry metadata

In practice, Prefit assumes the project already points at a usable
cluster-derived component set and an active template. The supporting
applications prepare those upstream inputs before the main SAXS UI turns them
into a model preview.

In plain language, Prefit is the "does this model make sense before I run a
heavier fit?" tab.

!!! info "Image placeholder"
Add a screenshot of the **SAXS Prefit** tab showing the main plot, the
parameter table, and the controls a first-time user should inspect before
moving to DREAM.

## When to use Prefit first

Use Prefit before DREAM when you need to:

- confirm that the built SAXS components load cleanly
- sanity-check the overall shape and scale of the model against the data
- decide whether a template or q-range choice is obviously wrong
- prepare geometry-aware metadata before a longer Bayesian run

## What you can do in Prefit

From the current UI implementation, Prefit supports:

- updating the model without leaving the tab
- autoscaling the plot
- toggling experimental, model, and solvent traces
- viewing fit metrics directly on the plot
- exporting plot data
- saving and restoring Prefit state
- saving a **Best Prefit** parameter preset
- using embedded solution-scattering estimators for solute volume fraction,
  solvent attenuation scaling, and fluorescence-background screening

The parameter table also supports lightweight Artemis-style expression modes in
the **Value** column:

- if **Value** is a math expression and **Vary** is enabled, Prefit treats the
  expression as a `guess`-style initial-value seed. The expression is resolved
  into the current numeric starting value, but the parameter can still refine
  inside **Min** / **Max**.
- if **Value** is a math expression and **Vary** is disabled, Prefit treats the
  expression as a `def`-style dependent parameter. In that mode the parameter
  follows the expression during evaluation and fitting, and its own **Min** /
  **Max** are ignored.

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

## Solution scattering estimators

Prefit now includes an embedded **Solution Scattering Estimators** section, and
the same calculations are also available from the **Tools** menu as:

- **Open Volume Fraction Estimate**
- **Open Attenuation Estimate**
- **Open Fluorescence Estimate**

All three calculators share the same composition inputs:

- solution density
- solute and solvent stoichiometries
- component molar masses
- component densities for physical volume-fraction estimates and diagnostics
- beam energy
- capillary size and geometry
- beam footprint and beam profile

The current implementation assumes a centered beam footprint and a uniform beam
profile. Flat-plate geometry uses a constant path length, while cylindrical
geometry averages across the illuminated chord lengths of the capillary.

### Software and data sources

The solution-composition bookkeeping is handled by SAXSShell's internal
solution-property helpers. Attenuation and fluorescence quantities are then
estimated from empirical formulas using `xraydb`, which exposes Elam-style
atomic edge, line, and fluorescence-yield data together with NIST-style mass
attenuation calculations.

In practical terms, SAXSShell combines:

- solution masses, densities, and stoichiometries from the Prefit widget
- linear attenuation coefficients derived from empirical formulas
- edge-resolved fluorescence yields and line families from `xraydb`
- beam-path averages defined by the selected capillary geometry
- molarity and volume-percent conventions matching
  [OpenStax Chemistry 2e](https://openstax.org/books/chemistry-2e/pages/3-3-molarity)
  and
  [Chemistry LibreTexts](<https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Chemistry_for_Allied_Health_(Soult)/08%3A_Properties_of_Solutions/8.01%3A_Concentrations_of_Solutions>)

### Physical solute-associated volume fraction

For the physical solute-associated volume-fraction estimate, SAXSShell uses the
measured solution volume together with the solute density:

$$
c_{\mathrm{solute}} = \frac{m_{\mathrm{solute}}}{V_{\mathrm{solution}}},
\qquad
\bar{v}_{\mathrm{solute}} \approx \frac{1}{\rho_{\mathrm{solute}}},
\qquad
\phi_{\mathrm{phys}} \approx c_{\mathrm{solute}} \bar{v}_{\mathrm{solute}}.
$$

The solvent fraction is then reported as:

$$
\phi_{\mathrm{solvent,phys}} = 1 - \phi_{\mathrm{phys}}.
$$

This is closer to the concentration-plus-specific-volume logic commonly used in
solution SAXS than the older additive-volume estimate
$V_{\mathrm{solute}} / (V_{\mathrm{solute}} + V_{\mathrm{solvent}})$.

For `molarity_per_liter` inputs, SAXSShell uses a one-liter final-solution
basis: molarity defines moles of solute per liter of solution, not per liter of
neat solvent. If a solute density is available, it is the primary
volume-fraction route:

$$
V_{\mathrm{solute}} =
\frac{n_{\mathrm{solute}} M_{\mathrm{solute}}}
{\rho_{\mathrm{solute}}},
\qquad
\phi_{\mathrm{phys}} =
\frac{V_{\mathrm{solute}}}{1000\ \mathrm{cm^3}}.
$$

For the built-in `PbI2 - DMF - 0.49 M` preset, this gives
\(m*{\mathrm{PbI_2}} = 0.49 \times 461.01 = 225.8949\ \mathrm{g}\).
Using \(\rho*{\mathrm{PbI*2}} = 6.16\ \mathrm{g/mL}\) gives
\(V*{\mathrm{PbI*2}} = 36.671\ \mathrm{cm^3}\) and
\(\phi*{\mathrm{phys}} = 0.036671\) (3.6671%).

The solvent density is still useful as a consistency diagnostic. With the same
preset, \(\rho*{\mathrm{solution}} = 1.144\ \mathrm{g/mL}\) and
\(\rho*{\mathrm{DMF}} = 0.944\ \mathrm{g/mL}\) imply a neat-solvent diagnostic
volume of \(918.1051 / 0.944 = 972.569\ \mathrm{cm^3}\), and an additive
component volume ratio of 1.009240. If no solute density is supplied,
SAXSShell can fall back to solvent-density closure
\(V*{\mathrm{solute}} = V*{\mathrm{solution}} - V\_{\mathrm{solvent}}\), which
would give 0.027431 for this preset. That fallback is intentionally reported as
a different method because solution volumes are not generally additive.

The regression tests also include an independent volume-percent example from
Chemistry LibreTexts: 40 mL ethanol in 240 mL final solution is 16.7% v/v.
Converting that known composition to a molarity/density input recovers
\(\phi = 40 / 240\), which guards the calculator against confusing final
solution volume with solvent volume.

SAXSShell still prints this physical estimate in the output console for
reference. It is written to a Prefit parameter only when the active template
explicitly declares a physical volume-fraction target, such as `vol_frac` in
`pyDREAM MonoSQ Normalized (Scaled Solvent Weight)`. Older templates that expose
`phi_solute` / `phi_solvent` keep using the SAXS-effective ratio described
below.

### SAXS-effective interaction contrast ratio

The model-facing solute fraction now uses an energy-dependent
contrast-weighted interaction ratio. SAXSShell forms an effective forward
scattering-electron density proxy from the component formula, density, and the
real anomalous correction \(f'(E)\):

$$
\rho_{\mathrm{eff}}(E)
=
\rho_{\mathrm{mass}}
\frac{N_A}{M}
\sum_i n_i \left[ Z_i + f'_i(E) \right].
$$

Using the solute-solvent contrast
\(\Delta \rho*{\mathrm{eff}}(E) = \rho*{\mathrm{eff,solute}}(E) - \rho\_{\mathrm{eff,solvent}}(E)\),
SAXSShell defines a contrast-weight factor

$$
C(E)
=
\left(
\frac{\Delta \rho_{\mathrm{eff}}(E)}
{\rho_{\mathrm{eff,solvent}}(E)}
\right)^2,
$$

an effective solute interaction volume

$$
V_{\mathrm{solute}}^{\mathrm{eff}}(E)
=
C(E) \, V_{\mathrm{solute,phys}},
$$

and the model-facing SAXS ratio

$$
R_{\mathrm{saxs}}(E)
=
\frac{V_{\mathrm{solute}}^{\mathrm{eff}}(E)}
{V_{\mathrm{solute}}^{\mathrm{eff}}(E) + V_{\mathrm{solvent,phys}}}.
$$

This keeps the physical occupancy estimate visible, but it lets the model
scale the solute and solvent terms using a contrast-sensitive ratio at the
selected beam energy. If the active template exposes `phi_solute` or
`phi_solvent`, Prefit now writes `R_saxs(E)` or its complement into that
parameter and sets `vary = off`.

### Attenuation and solvent scattering scale

For attenuation, SAXSShell forms the sample and neat-solvent linear attenuation
coefficients from concentration-weighted mass attenuation coefficients:

$$
\mu_{\mathrm{sample}}(E)
= c_{\mathrm{solute}} \left(\frac{\mu}{\rho}\right)_{\mathrm{solute}}(E)
+ c_{\mathrm{solvent}} \left(\frac{\mu}{\rho}\right)_{\mathrm{solvent}}(E),
$$

$$
\mu_{\mathrm{neat}}(E)
= \rho_{\mathrm{solvent}}
\left(\frac{\mu}{\rho}\right)_{\mathrm{solvent}}(E).
$$

For a path length $L$, the transmission model is:

$$
T(E, L) = e^{-\mu(E)L}.
$$

To estimate how much the neat-solvent reference should be scaled down before it
represents the solvent fraction inside the sample, SAXSShell compares
beam-profile-averaged scattering weights:

$$
w_{\mathrm{solv}}
=
\frac{
c_{\mathrm{solvent}}
\left\langle L e^{-\mu_{\mathrm{sample}} L} \right\rangle
}{
\rho_{\mathrm{solvent}}
\left\langle L e^{-\mu_{\mathrm{neat}} L} \right\rangle
}.
$$

Here $\langle \cdots \rangle$ denotes the path-length average across the
illuminated capillary cross section. This produces a solvent contribution scale
factor that answers the practical SAXS question, "how much more scattering
intensity does the neat solvent have than the solvent fraction inside the real
sample?"

Prefit writes these estimates into different model parameters depending on the
active template's declared convention.

For split-fraction templates such as the Poly LMA hard-sphere templates,
Prefit writes the attenuation factor above into `solvent_scale` and writes
`R_saxs(E)` or its solvent complement into the model-facing fraction parameter
(`phi_solute` / `phi_solvent`). The solvent term therefore uses the attenuation
scale together with the SAXS-effective solvent fraction:
\(w*{\mathrm{solv}} (1 - R*{\mathrm{saxs}}) I\_{\mathrm{solv}}(q)\).

For the original `pyDREAM MonoSQ Normalized` template, Prefit preserves the
historical single-parameter convention. There is no automatic `vol_frac` target,
and `solv_w` receives the combined solvent-background multiplier

$$
w_{\mathrm{model}} = \left(1 - R_{\mathrm{saxs}}(E)\right) w_{\mathrm{solv}}
$$

directly. In that original MonoSQ model, the solvent branch is added after the
global `scale`, so `solv_w` may also absorb any intensity-unit mismatch between
the experimental solvent blank and the scaled MD-derived model.

For `pyDREAM MonoSQ Normalized (Scaled Solvent Weight)`, Prefit uses the same
combined solvent-background multiplier for `solv_w`, but the template places
that weighted solvent trace inside the global scale. It also declares `vol_frac`
as a calculator target, so Prefit writes the physical solute-associated volume
fraction into `vol_frac` while keeping `solv_w` as the solvent-background
multiplier.

Prefit preserves user-edited bounds for `solv_w` and `solvent_scale` when
resetting states and when handing a project to DREAM batch processing. The
preloaded templates still start with conservative solvent-weight defaults, but
they do not force the solvent multiplier back into `[0, 1]` after you expand
the range.

That scaled-solvent MonoSQ template also asks Prefit to autoscale on load. If
experimental data are present and the project does not already have a saved Best
Prefit or current Prefit state for that template, Prefit applies the autoscale
estimate immediately and narrows the `scale` and `offset` limits around the
computed values.

### Fluorescence background proxy

The fluorescence estimator is intentionally a screening calculation rather than
a full transport simulation. It starts from the sample photoelectric
attenuation at the incident energy $E_0$, partitions that absorption across
accessible edges using edge jump ratios, and then applies fluorescence yields
and line-family branching:

$$
Y^{(1)}_{e,\ell}
\propto
\mu^{\mathrm{photo}}_{e,\mathrm{edge}}(E_0)\,
\omega_{e,\mathrm{edge}}\,
p_{e,\ell}\,
\left\langle
\mathcal{I}(L; \mu_{\mathrm{in}}, \mu_{\mathrm{out}})
\right\rangle.
$$

In this expression:

- $e$ is the emitting element
- $\ell$ is the emitted line family
- $\omega$ is the fluorescence yield
- $p$ is the line branching probability
- $\mathcal{I}$ is the path-integrated incident/escape attenuation term

SAXSShell then adds a first-order secondary-fluorescence pass by allowing the
primary fluorescent photons to be reabsorbed once and re-emitted before
escaping the sample. This is useful for ranking samples by expected
fluorescence-background severity, but it is not a Monte Carlo X-ray transport
calculation and should not be treated as a detector-absolute prediction.

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

That same practical issue becomes even more important when SAXSShell is fitting
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

In SAXSShell, `scale` is therefore kept in most templates as a practical way to
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

## Related pages

- [Project Configuration](project-configuration.md)
- [Cluster Extraction](cluster-extraction.md)
- [Template System](template-system.md)
- [Pre-loaded SAXS Models](preloaded-saxs-models.md)
- [LMFit Workflow](lmfit-workflow.md)
- [pyDREAM Workflow](pydream-workflow.md)

## References

- [SasView sphere model: absolute-scale interpretation of `scale`, explicit contrast term, and flat `background`.](https://www.sasview.org/docs/user/models/sphere.html)
- [Pedersen JS. _Analysis of small-angle scattering data from colloids and polymer solutions: modeling and least-squares fitting_. Advances in Colloid and Interface Science (1997).](<https://doi.org/10.1016/S0001-8686(97)00312-6>)
- [SasView power-law model: example of a flat additive background term.](https://www.sasview.org/docs/user/models/power_law.html)
- [Schneidman-Duhovny D, Hammel M, Tainer JA, Sali A. _Accurate SAXS profile computation and its assessment by contrast variation experiments_. Biophysical Journal (2013).](https://pubmed.ncbi.nlm.nih.gov/23972848/)
- [Henriques J, Arleth L, Lindorff-Larsen K, Skepö M. _On the Calculation of SAXS Profiles of Folded and Intrinsically Disordered Proteins from Computer Simulations_. Journal of Molecular Biology (2018).](https://pubmed.ncbi.nlm.nih.gov/29548755/)
- [Edwards-Gayle CJC, Khunti N, Hamley IW, Inoue K, Cowieson N, Rambo RP. _Design of a multipurpose sample cell holder for the Diamond Light Source high-throughput SAXS beamline B21_. Journal of Synchrotron Radiation (2021).](https://pmc.ncbi.nlm.nih.gov/articles/PMC7842227/)
- [Hajizadeh N, Franke D, Jeffries CM, Svergun DI. _Consensus Bayesian assessment of protein molecular mass from solution X-ray scattering data_. Scientific Reports (2018).](https://www.nature.com/articles/s41598-018-25355-2)
- [Hubbell JH. _Photon Cross Sections, Attenuation Coefficients, and Energy Absorption Coefficients from 10 keV to 100 GeV_. NBS NSRDS 29.](https://doi.org/10.6028/NBS.NSRDS.29)
- [XrayDB Python reference: attenuation, edge, and fluorescence-yield APIs used by SAXSShell.](https://scikit-beam.github.io/XrayDB/python.html)
- [Elam WT, Ravel BD, Sieber JR. _A new atomic database for X-ray spectroscopic calculations_. Radiation Physics and Chemistry (2002).](<https://doi.org/10.1016/S0969-806X(01)00327-4>)
- [Roter B, et al. Discussion of edge jump ratios and fluorescence forward modeling.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12871215/)
- [Trevorah RM, et al. Discussion of self-absorption and fluorescence re-absorption corrections.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6608621/)
