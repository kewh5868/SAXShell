# Project Setup

The **Project Setup** tab is where a SAXS project becomes concrete. It is the
place to connect experimental data, cluster-derived scattering components, prior
weights, and the active model template.

## What lives here

The current UI code shows Project Setup as the first tab in the SAXS
application. This is where you typically:

- choose or confirm the project directory
- select experimental SAXS data
- point the project at a cluster folder
- select a model template
- build or refresh prior-weight and component metadata
- preview histogram and distribution plots before moving to Prefit

## Typical setup order

1. Open the SAXS application with `saxs ui`.
2. Create a new project or load an existing project directory.
3. Select the experimental dataset.
4. Select the cluster folder you want to model.
5. Choose the template you want to use.
6. Build the prior-weight and SAXS component inputs.
7. Review the project summary and prior plots.
8. Move to **SAXS Prefit** once the component build is ready.

## Model and Build section

The Project Setup tab now also includes an **Install Model** action. This is
for templates authored as Python model files that pass the repository's
template-validation rules.

The install flow collects:

- a model name
- a `.py` template file
- a description used to generate paired JSON metadata

Successful installs become available to future projects from the same template
directory.

## Template selection

Template selection drives what the rest of the workflow allows.

Examples from the current codebase include:

- normalized monodisperse templates
- poly-LMA hard-sphere templates
- mixed approximate sphere/ellipsoid templates
- deprecated templates, which are hidden by default but still load for older
  projects

## Practical advice

- Finish the project-build steps before judging Prefit behavior. Many Prefit
  features depend on the component list and template selected here.
- If you are experimenting with custom templates, validate and install them
  before building a new project around them.
- If a template enables cluster geometry metadata, that capability shows up
  later in the Prefit workflow rather than in Project Setup itself.

## Related pages

- [Project Configuration](../user-guide/project-configuration.md)
- [Template System](../user-guide/template-system.md)
- [SAXS Prefit](../user-guide/saxs-prefit.md)
