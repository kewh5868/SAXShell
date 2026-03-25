# Template System

The SAXS workflow is template-driven. Templates define the model behavior, the
parameter surface, and any additional runtime inputs required by Prefit or
pyDREAM.

## Template pair structure

The current system uses a Python model file and a paired JSON metadata file.

In practice, the Python file defines the callable behavior, while the JSON file
supplies user-facing metadata such as display names, descriptions, and template
capabilities.

## Bundled templates

The repository currently includes bundled templates such as:

- normalized monodisperse workflows
- poly-LMA hard-sphere workflows
- approximate mixed sphere/ellipsoid workflows

Some older templates now live in a `_deprecated` subfolder. They are hidden by
default in template dropdowns, but older projects can still load them.

## Installing templates from the UI

The Project Setup tab now includes **Install Model**. The install flow validates
candidate templates before copying them into the repository's template area.

Validation currently checks for things such as:

- the expected template header directives
- lmfit and pyDREAM callables
- finite log-likelihood behavior
- geometry-capability consistency when geometry constraints are declared

## Geometry-aware templates

Templates can declare capabilities such as:

- support for cluster geometry metadata
- allowed shape approximations
- runtime metadata bindings

These capabilities directly control what the Prefit geometry table allows.

## Common model parameters: `scale` and `offset`

Many bundled templates include a global `scale` parameter and a constant
`offset` parameter.

This is intentional.

For SAXS models, the measured intensity depends on more than just shape. It
also depends on contrast and normalization terms. Even when data have been
placed on an absolute intensity scale, a free `scale` parameter can still be
scientifically reasonable if the effective contrast between the simulated
solute, its solvation layer, and the surrounding medium is not known exactly.

This is especially relevant for structure-derived scattering workflows, where
published SAXS studies show that uncertainties in hydration-layer contrast can
materially change the calculated profile.

Likewise, `offset` is included because a flat residual background is common in
real SAXS data reduction. In practice this can represent imperfect background
subtraction or residual background from the sample environment, including
parasitic scattering from windows or holders.

## Scientific note on current poly-LMA variants

The repository now distinguishes between:

- a strict hard-sphere template for sphere-only geometry
- an approximate mixed sphere/ellipsoid template that maps nonspherical rows to
  equivalent-volume spheres before evaluating a hard-sphere structure factor

That split matters because the mixed model is an approximation, not a full
anisotropic hard-ellipsoid closure.

## References

- [SasView sphere model documentation.](https://www.sasview.org/docs/user/models/sphere.html)
- [Henriques J, Arleth L, Lindorff-Larsen K, Skepö M. _On the Calculation of SAXS Profiles of Folded and Intrinsically Disordered Proteins from Computer Simulations_. Journal of Molecular Biology (2018).](https://pubmed.ncbi.nlm.nih.gov/29548755/)
- [Edwards-Gayle CJC, Khunti N, Hamley IW, Inoue K, Cowieson N, Rambo RP. _Design of a multipurpose sample cell holder for the Diamond Light Source high-throughput SAXS beamline B21_. Journal of Synchrotron Radiation (2021).](https://pmc.ncbi.nlm.nih.gov/articles/PMC7842227/)

## CLI support

The `saxs` CLI includes template management commands:

```bash
saxs templates
saxs templates validate path/to/template.py
saxs templates install path/to/template.py
```
