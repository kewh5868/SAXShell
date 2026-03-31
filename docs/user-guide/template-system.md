# Template System

The SAXS workflow is template-driven. Templates define the model behavior, the
parameter surface, and any additional runtime inputs required by Prefit or
pyDREAM.

## Template pair structure

The current system uses a Python model file and a paired JSON metadata file.

In practice, the Python file defines the callable behavior, while the JSON file
supplies user-facing metadata such as display names, descriptions, and template
capabilities.

## Python Header Syntax

Every installable template starts with a small header in the Python file. The
installer and loader read these lines before importing the module.

The required directives are:

- `# model_lmfit: <callable_name>`
- `# model_pydream: <callable_name>`
- `# inputs_lmfit: q, solvent_data, model_data, ..., params`
- `# inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, ..., params`
- `# param_columns: Structure, Motif, Param, Value, Vary, Min, Max`

Optional directives currently include:

- `# cluster_geometry_metadata: true`

Static parameter rows are declared with repeated `# param:` lines:

```python
# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_candidate
# inputs_lmfit: q, solvent_data, model_data, effective_radii, params
# inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, effective_radii, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
# cluster_geometry_metadata: true
#
# param: phi_solute,0.02,True,0.0,0.5
# param: phi_int,0.02,True,0.0,0.4
# param: scale,1.0,True,1e-8,1e8
# param: offset,0.0,True,-1e6,1e6
# param: log_sigma,-9.21,True,-20.0,5.0
```

Each `# param:` row must contain exactly five values:

```text
name, initial_value, vary, minimum, maximum
```

For example:

- `scale,1.0,True,1e-8,1e8`
- `offset,0.0,True,-1e6,1e6`

Important practical rules:

- the callable named by `model_lmfit` must exist in the module
- the callable named by `model_pydream` must exist in the module
- template-defined parameter names cannot reuse generated weight names like `w0`, `w1`, and so on
- duplicate parameter names are rejected
- `inputs_lmfit` and `inputs_pydream` must match the runtime contract the template actually expects

## JSON Metadata Format

The paired JSON file supplies the user-facing template metadata.

At minimum, it should contain:

```json
{
  "display_name": "My Template Name",
  "description": "A user-facing description of the template."
}
```

If no JSON file is present, SAXSShell can still load the Python model, but the
UI falls back to a generated display name and a generic description. The
installer allows this, but it will report a warning because the template will
look unfinished in the GUI.

## What The Template Installer Validates

The **Install Model** flow does more than check that a file imports. It stages
the candidate template in a temporary directory and validates it against the
same workflow objects used by the real application.

The current validator checks:

1. **Template header and metadata load**
   - the required header directives are present
   - the JSON metadata, if supplied, loads successfully
   - the display name and description are populated
2. **Parameter-definition sanity**
   - no duplicate parameter names
   - no template-defined names that collide with generated weight names such as `w0`
3. **Module import and callable resolution**
   - the lmfit callable exists and is callable
   - the pyDREAM callable exists and is callable
4. **Function contract validation on synthetic data**
   - the lmfit callable is run on a small synthetic `q` grid
   - it must return an array with the same shape as `q`
   - that array must be finite
   - the pyDREAM callable is run with synthetic globals and synthetic parameters
   - it must return a finite scalar log-likelihood
5. **Prefit workflow compatibility**
   - the template must evaluate successfully through `SAXSPrefitWorkflow`
   - the template must survive a short `run_fit(...)` call
6. **Cluster geometry constraint validation**, when enabled
   - geometry rows must load
   - allowed `S.F. Approx.` values must remain within the declared allowed set
   - disallowed approximations must normalize back to an allowed choice
   - dynamic geometry parameter rows must appear with the expected names
   - generated geometry parameters must default to `vary = False`
7. **DREAM runtime compatibility**
   - SAXSShell writes a real DREAM runtime bundle
   - it runs the bundle
   - the sampled-parameter and log-posterior arrays must have the expected shapes
   - sampled parameters must be finite
   - log-posterior arrays must not contain `NaN` or `+inf`

That means the installer is testing both the template interface and the way the
template behaves inside the real Prefit and DREAM application paths.

## Bundled templates

The repository currently includes bundled templates such as:

- normalized monodisperse workflows
- poly-LMA hard-sphere workflows
- approximate mixed sphere/ellipsoid workflows

For the bundled model equations, variable definitions, and literature links,
see [Pre-loaded SAXS Models](preloaded-saxs-models.md).

Some older templates now live in a `_deprecated` subfolder. They are hidden by
default in template dropdowns, but older projects can still load them.

## Installing templates from the UI

The Project Setup tab now includes **Install Model**. The install flow validates
candidate templates before copying them into the repository's template area.

If you provide only a Python file plus a model name and description in the UI,
SAXSShell generates the paired JSON metadata automatically before running the
validation pass.

## Geometry-aware templates

Templates can declare capabilities such as:

- support for cluster geometry metadata
- allowed shape approximations
- runtime metadata bindings

These capabilities directly control what the Prefit geometry table allows.

### Cluster Geometry Metadata Template Syntax

Geometry-aware templates need **both** the Python header directive and the JSON
capability block.

In the Python file:

```python
# cluster_geometry_metadata: true
```

In the JSON metadata:

```json
{
  "capabilities": {
    "cluster_geometry_metadata": {
      "supported": true,
      "mapping_target": "component_weights",
      "metadata_fields": [
        "effective_radius",
        "structure_factor_recommendation",
        "avg_size_metric",
        "anisotropy_metric",
        "notes"
      ],
      "runtime_bindings": {
        "effective_radii": "effective_radius"
      },
      "allowed_sf_approximations": ["sphere", "ellipsoid"],
      "dynamic_parameters": true,
      "sphere_parameter_prefix": "r_eff",
      "ellipsoid_parameter_prefixes": ["a_eff", "b_eff", "c_eff"]
    }
  }
}
```

The important fields are:

- `supported`
  - turns geometry support on
- `mapping_target`
  - currently only `component_weights` is supported
- `metadata_fields`
  - the set of row fields the template declares as meaningful
- `runtime_bindings`
  - maps runtime variable names to one of the declared metadata fields
- `allowed_sf_approximations`
  - currently must be drawn from `sphere` and `ellipsoid`
- `dynamic_parameters`
  - tells SAXSShell to generate geometry-derived parameter rows in Prefit/DREAM
- `sphere_parameter_prefix`
  - prefix for generated sphere parameters such as `r_eff_w0`
- `ellipsoid_parameter_prefixes`
  - three prefixes for ellipsoid semiaxis parameters such as `a_eff_w0`, `b_eff_w0`, `c_eff_w0`

Two contract details matter here:

- every runtime binding name must also appear in both `inputs_lmfit` and `inputs_pydream`
- every `runtime_bindings` target field must also be listed in `metadata_fields`

If you miss either of those, the template loader or installer will reject the
template before installation.

### Designing Geometry-Aware Templates

When making a geometry-aware template, decide early whether geometry is:

- a required runtime input only
- a required runtime input plus generated dynamic parameters
- sphere-only
- mixed sphere/ellipsoid

In practice:

- use `allowed_sf_approximations: ["sphere"]` for strict sphere-only models
- use `["sphere", "ellipsoid"]` only when your forward model has a clear approximation path for both
- enable `dynamic_parameters` if the active geometry choice should appear as explicit Prefit/DREAM parameters

Remember that the Prefit table is only the UI layer. The template still has to
interpret the active geometry physically. For example, the mixed Poly-LMA
approximate template maps ellipsoid semiaxes onto an equivalent-volume sphere
before evaluating a hard-sphere structure factor. That is a valid design only
because the template explicitly documents that approximation.

## Designing The pyDREAM Log-Likelihood Function

The pyDREAM callable should return a **single scalar log-likelihood** from the
current active parameter vector. In SAXSShell, the runtime bundle injects the
arrays declared in `inputs_pydream` as module-level globals, so most templates
follow this pattern:

```python
def log_likelihood_candidate(params):
    weight = float(params[0])
    scale = float(params[1])
    offset = float(params[2])
    log_sigma = float(params[3])

    model = lmfit_model_profile(
        q_values,
        solvent_intensities,
        [weight * theoretical_intensities[0]],
        scale=scale,
        offset=offset,
    )

    sigma = np.exp(log_sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return -np.inf

    residuals = np.asarray(experimental_intensities, dtype=float) - model
    return float(
        -0.5 * np.mean((residuals / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
    )
```

### Design Considerations For A Good Log-Likelihood

1. **Return a scalar**
   - the validator expects a single finite number, not a vector
2. **Keep invalid proposals cheap**
   - if a parameter proposal is physically impossible, return `-np.inf`
   - this is usually better than raising an exception inside the sampler
3. **Avoid `NaN` and `+inf`**
   - the validator and DREAM runtime both reject those
4. **Keep `sigma` positive**
   - fitting `log_sigma` and converting with `sigma = exp(log_sigma)` is often the cleanest way to enforce positivity
5. **Match shapes exactly**
   - your model output should align with `experimental_intensities` on the same `q_values` grid
6. **Normalize consciously**
   - some bundled templates divide the log-likelihood by the number of `q` points to make runs more comparable across different fitted ranges
7. **Be explicit about bounds**
   - if `phi`, `scale`, weights, or radii have a physical domain, guard it explicitly in the likelihood
8. **Use stable internal helpers**
   - keep the forward model in a separate lmfit-style helper so Prefit and DREAM evaluate the same physics
9. **Think about noise model assumptions**
   - Gaussian residuals are common, but only appropriate if that matches your practical error model
10. **Document approximations**

- if the likelihood depends on an equivalent-sphere or other reduced representation, say so clearly in the template comments and JSON description

### Practical Advice

When a candidate template fails installation, the most common causes are:

- missing or misspelled header directives
- callable names in the header that do not exist in the module
- lmfit functions that return the wrong array shape
- pyDREAM functions that rely on undeclared globals
- log-likelihood functions that raise exceptions instead of returning `-np.inf` for invalid proposals
- geometry runtime bindings declared in JSON but omitted from `inputs_lmfit` or `inputs_pydream`

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

For the current poly-LMA solvent-subtraction workflows, the repository also
distinguishes between:

- a physical bulk-density solute-associated volume fraction reported for reference
- a SAXS-effective interaction ratio used for the model-facing `phi_solute` /
  `phi_solvent` default

That distinction matters because the solvent-background subtraction is carried
by both the solute/solvent split and the explicit solvent term. The model-facing
split therefore follows the contrast-weighted SAXS interaction estimate, while
the attenuation term stays in `solvent_scale`.

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

The SAXS CLI includes template management commands through the installed
umbrella command:

```bash
saxshell saxs templates
saxshell saxs templates validate path/to/template.py
saxshell saxs templates install path/to/template.py
```

From a source checkout, use the module directly:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs templates
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs templates validate path/to/template.py
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs templates install path/to/template.py
```
