# Project Setup

The **Project Setup** tab is where a SAXS project becomes a reusable computed
distribution. This is the point where you define the Project Setup snapshot,
optionally compute Debye-Waller factors, and decide how SAXS components should
be built for the active modeling branch.

## What lives here

The current UI code shows Project Setup as the first tab in the SAXS
application. This is where you typically:

- choose or confirm the project directory
- select experimental and optional solvent SAXS data
- point the project at frames, PDB structures, and clusters folders
- choose a model template
- set the q-range, grid behavior, recognized elements, and excluded elements
- create or load computed distributions
- optionally compute project-backed Debye-Waller factors
- build SAXS components with the selected component-build mode
- preview component traces and prior histograms before moving to Prefit

## Typical setup order

1. Open the SAXS application with `saxshell saxs ui` or
   `PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs ui`.
2. Create a new project or load an existing project directory.
3. Select the experimental dataset and the cluster folder you want to model.
4. Choose the template, q-range, grid mode, and excluded elements.
5. Pick the **Component build mode** for the current modeling branch.
6. Click **Create Computed Distribution**.
7. Optionally click **Compute Debye-Waller Factors** if the active clusters are
   PDB files and you want saved disorder terms for later workflows.
8. Click **Build SAXS Components**.
9. Review the component preview, cluster table, and prior histogram preview.
10. Move to **SAXS Prefit** once the active distribution has the component
    traces you want to fit.

## Computed distributions

The Project Setup tab now makes computed distributions explicit.

**Create Computed Distribution** stores the current Project Setup snapshot and
generates the matching prior-weight artifacts. The computed-distribution
dropdown then lets you reload any saved distribution for the same project, and
the **Active Computed Distribution** panel summarizes the saved state and
artifact readiness for the selected one.

From the current UI and project-manager behavior, a computed distribution is
defined by the active Project Setup snapshot, including:

- selected template
- component build mode
- cluster source folder
- q-range and grid behavior
- excluded elements
- observed-only versus observed-plus-predicted structure weighting
- model-only mode

When experimental-grid mode is enabled, the experimental data source also feeds
into the saved distribution identity.

Because the component build mode is part of that identity, the same project can
hold multiple otherwise-similar distributions at once, for example one built
with `No Contrast (Debye)` and one built with
`Born Approximation (Average)`.

## Component build modes

The **Component build mode** dropdown controls what happens when you click
**Build SAXS Components**.

| Mode                             | What happens                                                                                                                                                                                                                                                                                                   |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **No Contrast (Debye)**          | The main SAXS UI runs the direct Debye component builder and saves the component traces into the active computed distribution.                                                                                                                                                                                 |
| **Contrast (Debye)**             | SAXSShell opens the linked **SAXS Contrast Mode** workflow so you can analyze representative structures, compute electron-density terms, and build contrast-aware Debye traces for that computed distribution.                                                                                                 |
| **Born Approximation (Average)** | SAXSShell opens the linked **Electron Density Mapping** workflow in computed-distribution mode so you can compute per-stoichiometry electron-density profiles, apply optional solvent subtraction, evaluate Fourier transforms, and then push the resulting Born-approximation components back into the model. |

## Debye-Waller factors

**Compute Debye-Waller Factors** is an optional linked step in Project Setup.

Important current behavior:

- the button stays disabled until a project is open and the active clusters
  folder resolves to readable `PDB` files only
- the linked Debye-Waller tool inherits the active project directory and
  clusters directory automatically
- the readiness indicator turns green when saved Debye-Waller results match the
  current PDB clusters folder
- if saved factors exist for a different clusters folder, the indicator stays
  off and the tooltip explains the mismatch

You do not need Debye-Waller factors to create a computed distribution, but the
tool is intended to be run before component building when you plan to reuse
those saved disorder terms in later SAXSShell workflows.

## Model and Build section

Project Setup also includes **Install Custom Template**. This is for templates
authored as Python model files that pass the repository's template-validation
rules.

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

- Treat **Create Computed Distribution** as the point where you intentionally
  branch the project into a specific build configuration.
- Finish the Project Setup steps before judging Prefit behavior. Prefit and
  DREAM both depend on the active computed distribution and its saved
  component artifacts.
- If you switch build modes, q-range, template, or excluded elements, create a
  new computed distribution instead of assuming the previous saved state still
  applies.
- If you are experimenting with custom templates, validate and install them
  before building a new project around them.

## Related pages

- [Project Configuration](../user-guide/project-configuration.md)
- [GUI Overview](../user-guide/gui-overview.md)
- [Template System](../user-guide/template-system.md)
- [SAXS Prefit](../user-guide/saxs-prefit.md)
