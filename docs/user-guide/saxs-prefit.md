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
- component densities when the selected input mode requires them
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

SAXSShell still prints this physical estimate in the output console for
reference, but it is no longer written directly into the model-facing
`phi_solute` / `phi_solvent` defaults.

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

If the active template includes both a model-facing fraction parameter
(`phi_solute` / `phi_solvent`) and a solvent-weight parameter such as
`solvent_scale`, Prefit writes the attenuation factor above into
`solvent_scale` and uses `R_saxs(E)` for the fraction parameter. The solvent
term therefore becomes
\(w*{\mathrm{solv}} (1 - R*{\mathrm{saxs}}) I\_{\mathrm{solv}}(q)\).

If the template only exposes a single solvent-weight parameter such as
`solv_w`, Prefit writes the combined solvent-background multiplier

$$
w_{\mathrm{model}} = \left(1 - R_{\mathrm{saxs}}(E)\right) w_{\mathrm{solv}}
$$

into that parameter directly.

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

## TODO

TODO: re-check the consistency between the attenuation estimator output
(`solvent_scale` / `solv_w`) and the solvent contribution used by the model.
At least some current workflows appear to produce model-facing solvent weights
around `0.15` even when the imported solvent trace still looks about `100x`
too large by eye at the same q-position maxima. Revisit whether this comes
from a mismatch between:

- the attenuation-only estimate and the actual model solvent term
- split-fraction versus single-solvent-weight template conventions
- solvent blank versus sample normalization, transmission, thickness, or exposure
- partially pre-subtracted or otherwise inconsistently reduced solvent traces
- the possible need for a separate empirical solvent-normalization factor in
  addition to the attenuation scaling

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
