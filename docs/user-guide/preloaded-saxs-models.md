# Pre-loaded SAXS Models

This page documents the bundled SAXS templates that ship with SAXSShell. The
equations below describe the **implemented forward models in the repository**.
In a few places, SAXSShell combines MD-derived component mixtures with
literature structure-factor building blocks, so the exact code path is an
implementation of the cited ideas rather than a verbatim reproduction of a
single paper.

## Template Catalog

| Template file                                                  | GUI name                                                    | Status     | Model family                                  |
| -------------------------------------------------------------- | ----------------------------------------------------------- | ---------- | --------------------------------------------- |
| `template_pydream_monosq_normalized.py`                        | `pyDREAM MonoSQ Normalized`                                 | current    | MonoSQ hard-sphere                            |
| `template_pydream_monosq_normalized_scaled_solvent.py`         | `pyDREAM MonoSQ Normalized (Scaled Solvent Weight)`         | current    | MonoSQ hard-sphere with scale-coupled solvent |
| `template_pydream_charged_monosq_normalized_scaled_solvent.py` | `pyDREAM Charged MonoSQ Normalized (Scaled Solvent Weight)` | current    | MonoSQ charged hard-sphere RMSA               |
| `template_pydream_poly_lma_hs.py`                              | `pyDREAM Poly LMA Hard-Sphere`                              | current    | sphere-only Poly LMA hard-sphere              |
| `template_pydream_poly_lma_hs_mix_approx.py`                   | `pyDREAM Poly LMA Hard-Sphere/Ellipsoid Mix (Approx.)`      | current    | mixed-shape approximate Poly LMA hard-sphere  |
| `template_likelihood_monosq.py`                                | `MonoSQ Basic (archived)`                                   | archived   | MonoSQ hard-sphere                            |
| `template_pd_likelihood_monosq.py`                             | `MonoSQ PD (archived)`                                      | archived   | MonoSQ hard-sphere                            |
| `template_pd_likelihood_monosq_decoupled.py`                   | `MonoSQ Decoupled (archived)`                               | archived   | MonoSQ hard-sphere                            |
| `template_pydream_poly_lma_hs_legacy.py`                       | `pyDREAM Poly LMA Hard-Sphere (deprecated)`                 | deprecated | mixed-shape approximate Poly LMA hard-sphere  |

## Shared Notation

Across the bundled templates:

- \(q\) is the scattering vector magnitude.
- \(I_i(q)\) is the MD-derived SAXS profile for component \(i\).
- \(I\_{\mathrm{solv}}(q)\) is the solvent scattering trace.
- \(w_i\) is the raw weight assigned to component \(i\).
- \(S\_{\mathrm{HS}}(q; R, \phi)\) is the hard-sphere Percus-Yevick structure
  factor evaluated at effective radius \(R\) and packing term \(\phi\).
- \(S\_{\mathrm{RMSA}}(q)\) is the Hayter-Penfold rescaled mean spherical
  approximation charged-sphere structure factor.
- `scale` and `offset` are the global multiplicative and additive terms exposed
  in the Prefit parameter table.

## MonoSQ Hard-Sphere Family

Applies to:

- `template_pydream_monosq_normalized.py`
- `template_pydream_monosq_normalized_scaled_solvent.py`
- `template_likelihood_monosq.py`
- `template_pd_likelihood_monosq.py`
- `template_pd_likelihood_monosq_decoupled.py`

These templates treat the MD-derived component profiles as a weighted solute
mixture modulated by a **single** monodisperse hard-sphere structure factor.
They differ mainly in where the experimental solvent trace enters the scaled
model expression.

All MonoSQ templates start from the same solute branch:

$$
I_{\mathrm{mix}}(q) = \sum_{i=0}^{N-1} w_i I_i(q)
$$

$$
I_{\mathrm{solute}}(q) =
I_{\mathrm{mix}}(q) S_{\mathrm{HS}}(q; R_{\mathrm{eff}}, \phi_{\mathrm{vol}})
$$

### Current normalized MonoSQ

The original `pyDREAM MonoSQ Normalized` template keeps the historical
unscaled-solvent convention:

$$
I_{\mathrm{model}}(q) =
\mathrm{scale}\, I_{\mathrm{solute}}(q)
+ w_{\mathrm{solv}} I_{\mathrm{solv}}(q)
+ \mathrm{offset}.
$$

In this template, `scale` applies only to the MD-derived solute branch. The
solvent trace is added after the global scale, so `solv_w` must carry both the
physical solvent-background multiplier and any remaining intensity-unit
mismatch between the imported solvent data and the scaled MD model. This
preserves the behavior of existing projects, but it can make fitted `solv_w`
values look much smaller than a physical solvent volume fraction when the
experimental solvent trace is orders of magnitude larger than the model trace.

The solution-scattering calculator can still seed `solv_w` for this template
with the combined solvent-background multiplier. It does **not** seed `vol_frac`
for the original MonoSQ template; `vol_frac` remains a fitted hard-sphere
packing term.

### Scaled Solvent Weight MonoSQ

The `pyDREAM MonoSQ Normalized (Scaled Solvent Weight)` template keeps the same
MonoSQ solute branch and point-normalized likelihood, but moves the solvent
trace inside the global scale:

$$
I_{\mathrm{model}}(q) =
\mathrm{scale}
\left[
I_{\mathrm{solute}}(q)
+ w_{\mathrm{solv}} I_{\mathrm{solv}}(q)
\right]
+ \mathrm{offset}.
$$

Here `scale` applies to the combined solute-plus-solvent model. This makes
`solv_w` a model-facing solvent-background multiplier rather than a parameter
that also has to absorb the arbitrary MD-model intensity scale. In practice,
this is the safer MonoSQ starting point when the solvent blank intensity is
much larger than the unscaled MD component profiles.

This template declares calculator targets in its metadata:

- `vol_frac` receives the physical solute-associated volume fraction computed
  from the solution composition.
- `solv_w` receives the combined solvent-background multiplier from attenuation
  and SAXS-effective solvent contrast.

It also declares Prefit startup behavior. When experimental data are available
and there is no saved Best Prefit or current Prefit state for this template,
Prefit applies the autoscale recommendation as soon as the template loads. The
new `scale` and `offset` limits are centered around the autoscale result rather
than preserving the broad template-default ranges. The default `eff_r` starts at
3 A, the lower bound of the effective-radius search range.

Because the solvent branch is scale-coupled, Prefit's scale recommendation also
treats the solvent term as part of the scaled model instead of subtracting it as
an already-scaled background contribution.

### Charged Scaled Solvent MonoSQ

The `pyDREAM Charged MonoSQ Normalized (Scaled Solvent Weight)` template keeps
the scaled-solvent MonoSQ organization, but replaces the neutral
Percus-Yevick hard-sphere term with the Hayter-Penfold RMSA structure factor for
screened Coulomb repulsion between charged spheres.

The cluster-trace form-factor mixture is still

$$
I_{\mathrm{mix}}(q) = \sum_i w_i I_i(q).
$$

The charged solute branch is

$$
I_{\mathrm{solute}}(q) =
I_{\mathrm{mix}}(q)
S_{\mathrm{RMSA}}
\left(q; R_{\mathrm{eff}}, \phi, Z, T, c_{\mathrm{salt}}, \epsilon_r\right),
$$

and the full model follows the same scale-coupled solvent convention:

$$
I_{\mathrm{model}}(q) =
\mathrm{scale}
\left[
I_{\mathrm{solute}}(q)
+ w_{\mathrm{solv}} I_{\mathrm{solv}}(q)
\right]
+ \mathrm{offset}.
$$

Here \(Z\) is the charged-sphere charge in elementary-charge units,
\(T\) is the absolute temperature, \(c\_{\mathrm{salt}}\) is the molar
concentration of added 1:1 electrolyte, and \(\epsilon_r\) is the solvent
relative dielectric constant.

The implementation follows the SasView `hayter_msa` parameterization. The
template first converts the fitted parameters into SI-derived screening terms:

$$
\beta = \frac{1}{k_B T},
\qquad
\epsilon = \epsilon_r \epsilon_0,
\qquad
\sigma = 2R_{\mathrm{eff}}.
$$

For monovalent counterions and added 1:1 salt, the ionic-strength term used by
the RMSA kernel is

$$
I_{\mathrm{ion}} =
\frac{e^2}{2}
\left(
\frac{Z\phi}{V_p}
+ 2 N_A 10^3 c_{\mathrm{salt}}
\right),
$$

where \(V*p = 4\pi R*{\mathrm{eff}}^3 / 3\) after converting \(R\_{\mathrm{eff}}\)
to meters. The Debye-Huckel screening parameter is

$$
\kappa =
\sqrt{\frac{2\beta I_{\mathrm{ion}}}{\epsilon}}.
$$

The dimensionless contact-potential parameter passed into the
Hayter-Penfold coefficient calculation is

$$
\Gamma =
\frac{
\beta (Ze)^2
}{
\pi \epsilon \sigma (2 + \kappa\sigma)^2
}.
$$

The Hayter-Penfold rescaling solves for a rescaled volume fraction
\(\phi_s\), rescaled screening parameter \(\kappa_s\), and MSA coefficients
\(A, B, C, F, U, V\) that satisfy the Gillan contact condition. SAXSShell
then evaluates the same final structure-factor form used by SasView:

$$
S_{\mathrm{RMSA}}(q) =
\frac{1}{1 - 24\phi_s\,\mathcal{A}(q\sigma / s)},
$$

where \(s = (\phi / \phi_s)^{1/3}\) and
\(\mathcal{A}\) is the Hayter-Penfold Fourier-space coefficient expression.
The template includes the small-\(q\) Taylor branch used by SasView to avoid
rounding error near \(q\sigma / s = 0\).

This is a charged-particle model. `charge` is constrained to be positive and
bounded above by 200 e, matching the SasView stability guidance. For neutral
systems use one of the hard-sphere MonoSQ templates instead.

Like the scaled-solvent hard-sphere template, this charged template declares
calculator targets in its metadata:

- `vol_frac` receives the physical solute-associated volume fraction computed
  from the solution composition.
- `solv_w` receives the combined solvent-background multiplier from attenuation
  and SAXS-effective solvent contrast.
- The solvent contribution is marked as globally scaled, so Prefit's autoscale
  calculation treats the solvent branch as part of the model curve.

### Variables

| Symbol / parameter                            | Meaning in SAXSShell                                                                                                       |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| \(w_i\)                                       | generated component weight for cluster profile \(i\)                                                                       |
| \(w\_{\mathrm{solv}}\) / `solv_w`             | bounded solvent contribution weight                                                                                        |
| \(R\_{\mathrm{eff}}\) / `eff_r`               | effective hard-sphere radius used in `calc_monodisperse_sq(...)`; scaled-solvent MonoSQ defaults to 3 A                    |
| \(\phi\_{\mathrm{vol}}\) / `vol_frac`         | effective hard-sphere volume fraction inside the Percus-Yevick term                                                        |
| `scale`                                       | global intensity scale; original MonoSQ applies it only to solute, scaled-solvent MonoSQ applies it to solute plus solvent |
| `offset`                                      | constant additive background                                                                                               |
| \(Z\) / `charge`                              | charged-sphere charge in elementary-charge units for the charged RMSA template                                             |
| \(T\) / `temperature`                         | absolute temperature in kelvin for the charged RMSA Debye length calculation                                               |
| \(c\_{\mathrm{salt}}\) / `concentration_salt` | added 1:1 electrolyte concentration in mol/L for the charged RMSA template                                                 |
| \(\epsilon_r\) / `dielectconst`               | solvent relative dielectric constant for the charged RMSA template                                                         |

### Likelihood conventions

The current `pyDREAM MonoSQ Normalized` template uses a point-normalized
Gaussian log-likelihood with a fixed noise scale of \(10^{-4}\):

\[
\log \mathcal{L}_{\mathrm{norm}} =
\frac{1}{N_q}
\sum_{k=1}^{N*q}
\log \mathcal{N}
\left(
I*{\mathrm{exp}}(q*k) \mid I*{\mathrm{model}}(q_k), 10^{-4}
\right)
\]

The archived `MonoSQ Basic` template uses the same forward model but omits the
\(1/N_q\) normalization. The archived `MonoSQ Decoupled` template keeps the same
equation and simply factors the forward model into an intermediate helper
function before evaluating the likelihood.

### Literature

- J. K. Percus and G. J. Yevick, _Analysis of Classical Statistical Mechanics by Means of Collective Coordinates_,
  Phys. Rev. **110**, 1-13 (1958). <https://doi.org/10.1103/PhysRev.110.1>
- M. S. Wertheim, _Exact Solution of the Percus-Yevick Integral Equation for Hard Spheres_,
  Phys. Rev. Lett. **10**, 321-323 (1963). <https://doi.org/10.1103/PhysRevLett.10.321>
- J. S. Pedersen, _Analysis of small-angle scattering data from colloids and polymer solutions: modeling and least-squares fitting_,
  Adv. Colloid Interface Sci. **70**, 171-210 (1997). <https://doi.org/10.1016/S0001-8686(97)00312-6>
- J. B. Hayter and J. Penfold, Molecular Physics **42**, 109-118 (1981).
- J. P. Hansen and J. B. Hayter, Molecular Physics **46**, 651-656 (1982).
- SasView `hayter_msa` charged-sphere RMSA model documentation:
  <https://www.sasview.org/docs/user/models/hayter_msa.html>

## Poly LMA Hard-Sphere

Applies to:

- `template_pydream_poly_lma_hs.py`

This template uses a **discrete local-monodisperse-approximation-style**
cluster sum: each cluster profile keeps its own effective interaction radius,
but the cluster abundances are normalized internally before evaluating the
solute mixture.

\[
x_i = \frac{w_i}{\sum_j w_j},
\qquad
\sum_i x_i = 1
\]

\[
\begin{aligned}
I*{\mathrm{model}}(q) ={}&
\mathrm{scale}\,\phi*{\mathrm{solute}}
\sum*{i=0}^{N-1}
x_i I_i(q) S*{\mathrm{HS}}(q; R*i^{\mathrm{eff}}, \phi*{\mathrm{int}}) \\
&+ s*{\mathrm{solv}} (1-\phi*{\mathrm{solute}}) I\_{\mathrm{solv}}(q)

- \mathrm{offset}
  \end{aligned}
  \]

### Variables

| Symbol / parameter                         | Meaning in SAXSShell                                                                             |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| \(w_i\)                                    | raw cluster-abundance coefficient generated from the project component rows                      |
| \(x_i\)                                    | normalized abundance used internally by the model                                                |
| \(R_i^{\mathrm{eff}}\)                     | per-cluster effective interaction radius                                                         |
| `r_eff_wN`                                 | generated Prefit/DREAM radius parameter for cluster `wN` when sphere mode is active              |
| \(\phi\_{\mathrm{solute}}\) / `phi_solute` | SAXS-effective solute interaction ratio scaling the cluster contribution                         |
| \(\phi\_{\mathrm{int}}\) / `phi_int`       | interaction packing fraction used only inside the hard-sphere structure factor                   |
| \(s\_{\mathrm{solv}}\) / `solvent_scale`   | bounded attenuation solvent-scaling term, used together with the `phi_solute` solvent complement |
| `scale`                                    | solute intensity scale factor                                                                    |
| `offset`                                   | constant additive background                                                                     |
| \(\sigma = e^{\log \sigma}\) / `log_sigma` | Gaussian noise scale for the DREAM likelihood                                                    |

In the current implementation, \(R_i^{\mathrm{eff}}\) is taken from the
generated parameter `r_eff_wN` when that row exists. Otherwise, the template
falls back to the cluster-geometry metadata value supplied by Prefit.

### Likelihood convention

\[
\log \mathcal{L}_{\mathrm{norm}} =
\frac{1}{N_q}
\sum_{k=1}^{N*q}
\log \mathcal{N}
\left(
I*{\mathrm{exp}}(q*k) \mid I*{\mathrm{model}}(q_k), e^{\log \sigma}
\right)
\]

### Literature

- J. S. Pedersen, _Determination of size distributions from small-angle scattering data for systems with effective hard-sphere interactions_,
  J. Appl. Cryst. **27**, 595-608 (1994). <https://doi.org/10.1107/S0021889893013810>
- J. S. Pedersen, _Analysis of small-angle scattering data from colloids and polymer solutions: modeling and least-squares fitting_,
  Adv. Colloid Interface Sci. **70**, 171-210 (1997). <https://doi.org/10.1016/S0001-8686(97)00312-6>
- J. K. Percus and G. J. Yevick, _Analysis of Classical Statistical Mechanics by Means of Collective Coordinates_,
  Phys. Rev. **110**, 1-13 (1958). <https://doi.org/10.1103/PhysRev.110.1>
- M. S. Wertheim, _Exact Solution of the Percus-Yevick Integral Equation for Hard Spheres_,
  Phys. Rev. Lett. **10**, 321-323 (1963). <https://doi.org/10.1103/PhysRevLett.10.321>

## Poly LMA Hard-Sphere/Ellipsoid Mix (Approx.)

Applies to:

- `template_pydream_poly_lma_hs_mix_approx.py`
- `template_pydream_poly_lma_hs_legacy.py`

This template keeps the same cluster-summed hard-sphere equation as the
sphere-only Poly LMA model, but it allows Prefit geometry rows to be toggled
between sphere and ellipsoid approximations.

\[
\begin{aligned}
I*{\mathrm{model}}(q) ={}&
\mathrm{scale}\,\phi*{\mathrm{solute}}
\sum*{i=0}^{N-1}
x_i I_i(q) S*{\mathrm{HS}}(q; R*i^{\mathrm{eff}}, \phi*{\mathrm{int}}) \\
&+ s*{\mathrm{solv}} (1-\phi*{\mathrm{solute}}) I\_{\mathrm{solv}}(q)

- \mathrm{offset}
  \end{aligned}
  \]

The difference is how the effective interaction radius is resolved:

\[
R*i^{\mathrm{eff}} =
\begin{cases}
r*{\mathrm{eff},i}, & \text{if the component is treated as a sphere}, \\
\left(a_i b_i c_i\right)^{1/3}, & \text{if the component is treated as an ellipsoid}.
\end{cases}
\]

Here \(a_i\), \(b_i\), and \(c_i\) correspond to the generated semiaxis
parameters `a_eff_wN`, `b_eff_wN`, and `c_eff_wN`.

!!! warning "Approximation Scope"
This is a SAXSShell approximation, not an exact hard-ellipsoid
Percus-Yevick closure. Ellipsoid geometry is reduced to an
equivalent-volume sphere before the hard-sphere structure factor is
evaluated.

### Variables

The weight, solvent, `scale`, `offset`, `phi_solute`, `phi_int`, and
`log_sigma` terms are the same as in the sphere-only Poly LMA model. The extra
geometry-dependent parameters are:

| Symbol / parameter                 | Meaning in SAXSShell                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------------- |
| `r_eff_wN`                         | sphere radius parameter when the mapped component uses the sphere approximation          |
| `a_eff_wN`, `b_eff_wN`, `c_eff_wN` | ellipsoid semiaxis parameters when the mapped component uses the ellipsoid approximation |
| \(R_i^{\mathrm{eff}}\)             | effective radius actually passed into the hard-sphere structure factor                   |

### Literature

- S. Hansen, _Monte Carlo estimation of the structure factor for hard bodies in small-angle scattering_,
  J. Appl. Cryst. **45**, 381-388 (2012). <https://doi.org/10.1107/S0021889812009557>
- S. Hansen, _Approximation of the structure factor for nonspherical hard bodies using polydisperse spheres_,
  J. Appl. Cryst. **46**, 1008-1016 (2013). <https://doi.org/10.1107/S0021889813015392>
- J. S. Pedersen, _Determination of size distributions from small-angle scattering data for systems with effective hard-sphere interactions_,
  J. Appl. Cryst. **27**, 595-608 (1994). <https://doi.org/10.1107/S0021889893013810>

## Archived Template Notes

The archived templates are still loadable for older projects, but they map onto
the current model families as follows:

- `template_likelihood_monosq.py`: same MonoSQ forward model, legacy
  unnormalized Gaussian log-likelihood.
- `template_pd_likelihood_monosq.py`: same MonoSQ forward model, normalized
  Gaussian log-likelihood.
- `template_pd_likelihood_monosq_decoupled.py`: same MonoSQ forward model,
  normalized likelihood, explicit `model_monosq(...)` helper.
- `template_pydream_poly_lma_hs_legacy.py`: compatibility wrapper around the
  current mixed-shape approximate Poly LMA model.

## Related pages

- [Template System](template-system.md)
- [Project Configuration](project-configuration.md)
- [SAXS Prefit](saxs-prefit.md)
- [LMFit Workflow](lmfit-workflow.md)
- [pyDREAM Workflow](pydream-workflow.md)
