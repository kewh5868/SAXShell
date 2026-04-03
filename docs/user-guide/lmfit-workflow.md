# LMFit Workflow

In this repository, **Prefit** is the lmfit-facing side of the SAXS workflow.
It gives you a fast, editable model preview before you commit to a pyDREAM run.

That preview only becomes meaningful after the project has a template,
experimental data, and a cluster-derived component set from the supporting
applications.

## Role of lmfit in SAXSShell

The template system defines an lmfit-compatible profile function. Prefit then
uses that function together with the current parameter table to compute the
preview model.

## What the parameter table does

The parameter table in Prefit is the working parameter surface for the current
template. Depending on the template, it can include:

- component weight parameters such as `w0`, `w1`, ...
- global scale, offset, or solvent parameters
- dynamic geometry parameters derived from cluster geometry metadata

Geometry-aware templates can now regenerate these parameter rows when the
allowed shape per cluster changes.

## Best Prefit

The current Prefit workflow supports a **Best Prefit** preset. This is useful
when you want a stable parameter baseline for a specific template without
overwriting every saved Prefit snapshot.

## Save and restore

Use the save and restore actions to preserve:

- the current parameter table
- the cluster geometry table and active mode
- the current template runtime inputs

## When lmfit is enough

Prefit is often enough when you want:

- a sanity check on component construction
- rough parameter exploration
- a baseline fit before setting up priors for DREAM

If you need posterior uncertainty, parameter correlations, or a more formal
sampling workflow, move to the pyDREAM tab.

## Related pages

- [Project Configuration](project-configuration.md)
- [Cluster Extraction](cluster-extraction.md)
- [Template System](template-system.md)
- [Pre-loaded SAXS Models](preloaded-saxs-models.md)
- [SAXS Prefit](saxs-prefit.md)
- [pyDREAM Workflow](pydream-workflow.md)
