# Pre-loaded SAXS Models

This page documents the bundled SAXS templates that ship with SAXSShell. The
equations below describe the **implemented forward models in the repository**.
In a few places, SAXSShell combines MD-derived component mixtures with
literature structure-factor building blocks, so the exact code path is an
implementation of the cited ideas rather than a verbatim reproduction of a
single paper.

## Template Catalog

| Template file                                | GUI name                                               | Status     | Model family                                 |
| -------------------------------------------- | ------------------------------------------------------ | ---------- | -------------------------------------------- |
| `template_pydream_monosq_normalized.py`      | `pyDREAM MonoSQ Normalized`                            | current    | MonoSQ hard-sphere                           |
| `template_pydream_poly_lma_hs.py`            | `pyDREAM Poly LMA Hard-Sphere`                         | current    | sphere-only Poly LMA hard-sphere             |
| `template_pydream_poly_lma_hs_mix_approx.py` | `pyDREAM Poly LMA Hard-Sphere/Ellipsoid Mix (Approx.)` | current    | mixed-shape approximate Poly LMA hard-sphere |
| `template_likelihood_monosq.py`              | `MonoSQ Basic (archived)`                              | archived   | MonoSQ hard-sphere                           |
| `template_pd_likelihood_monosq.py`           | `MonoSQ PD (archived)`                                 | archived   | MonoSQ hard-sphere                           |
| `template_pd_likelihood_monosq_decoupled.py` | `MonoSQ Decoupled (archived)`                          | archived   | MonoSQ hard-sphere                           |
| `template_pydream_poly_lma_hs_legacy.py`     | `pyDREAM Poly LMA Hard-Sphere (deprecated)`            | deprecated | mixed-shape approximate Poly LMA hard-sphere |

## Shared Notation

Across the bundled templates:

- \(q\) is the scattering vector magnitude.
- \(I_i(q)\) is the MD-derived SAXS profile for component \(i\).
- \(I\_{\mathrm{solv}}(q)\) is the solvent scattering trace.
- \(w_i\) is the raw weight assigned to component \(i\).
- \(S\_{\mathrm{HS}}(q; R, \phi)\) is the hard-sphere Percus-Yevick structure
  factor evaluated at effective radius \(R\) and packing term \(\phi\).
- `scale` and `offset` are the global multiplicative and additive terms exposed
  in the Prefit parameter table.

## MonoSQ Hard-Sphere Family

Applies to:

- `template_pydream_monosq_normalized.py`
- `template_likelihood_monosq.py`
- `template_pd_likelihood_monosq.py`
- `template_pd_likelihood_monosq_decoupled.py`

These templates treat the MD-derived component profiles as a weighted solute
mixture modulated by a **single** monodisperse hard-sphere structure factor.

\[
I*{\mathrm{mix}}(q) = \sum*{i=0}^{N-1} w_i I_i(q)
\]

\[
I*{\mathrm{model}}(q) =
\mathrm{scale}\, I*{\mathrm{mix}}(q)\,
S*{\mathrm{HS}}(q; R*{\mathrm{eff}}, \phi\_{\mathrm{vol}})

- w*{\mathrm{solv}} I*{\mathrm{solv}}(q)
- \mathrm{offset}
  \]

### Variables

| Symbol / parameter                    | Meaning in SAXSShell                                                |
| ------------------------------------- | ------------------------------------------------------------------- |
| \(w_i\)                               | generated component weight for cluster profile \(i\)                |
| \(w\_{\mathrm{solv}}\) / `solv_w`     | bounded solvent contribution weight                                 |
| \(R\_{\mathrm{eff}}\) / `eff_r`       | effective hard-sphere radius used in `calc_monodisperse_sq(...)`    |
| \(\phi\_{\mathrm{vol}}\) / `vol_frac` | effective hard-sphere volume fraction inside the Percus-Yevick term |
| `scale`                               | solute intensity scale factor                                       |
| `offset`                              | constant additive background                                        |

### Likelihood conventions

The current `pyDREAM MonoSQ Normalized` template uses a point-normalized
Gaussian log-likelihood with a fixed noise scale of \(10^{-4}\):

\[
\log \mathcal{L}_{\mathrm{norm}} =
\frac{1}{N_q}
\sum_{k=1}^{N*q}
\log \mathcal{N}
\left(
I*{\exp}(q*k)\ \middle|\ I*{\mathrm{model}}(q_k), 10^{-4}
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
I*{\mathrm{model}}(q) =
\mathrm{scale}\,\phi*{\mathrm{solute}}
\sum*{i=0}^{N-1}
x_i I_i(q) S*{\mathrm{HS}}(q; R*i^{\mathrm{eff}}, \phi*{\mathrm{int}})

- s*{\mathrm{solv}} (1-\phi*{\mathrm{solute}}) I\_{\mathrm{solv}}(q)
- \mathrm{offset}
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
I*{\exp}(q*k)\ \middle|\ I*{\mathrm{model}}(q_k), e^{\log \sigma}
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
I*{\mathrm{model}}(q) =
\mathrm{scale}\,\phi*{\mathrm{solute}}
\sum*{i=0}^{N-1}
x_i I_i(q) S*{\mathrm{HS}}(q; R*i^{\mathrm{eff}}, \phi*{\mathrm{int}})

- s*{\mathrm{solv}} (1-\phi*{\mathrm{solute}}) I\_{\mathrm{solv}}(q)
- \mathrm{offset}
  \]

The difference is how the effective interaction radius is resolved:

\[
R*i^{\mathrm{eff}} =
\begin{cases}
r*{\mathrm{eff},i}, & \text{if the component is treated as a sphere} \\
\left(a_i b_i c_i\right)^{1/3}, & \text{if the component is treated as an ellipsoid}
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
