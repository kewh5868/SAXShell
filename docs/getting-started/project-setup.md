# Project Setup

The **Project Setup** tab is where a SAXS project becomes a reusable computed
distribution. This is the point where you define the Project Setup snapshot,
optionally compute Debye-Waller factors, and decide how SAXS components should
be built for the active modeling branch.

In plain language, this is where you tell SAXSShell what experimental SAXS data
you want to match, which simulation-derived cluster set should be compared
against that data, and which modeling branch should be saved for later fitting.

!!! info "Image placeholder"
Add a screenshot of the full **Project Setup** tab with a loaded project,
showing the project path, input selectors, computed-distribution controls,
and preview panels.

## What lives here

The current UI code shows Project Setup as the first tab in the SAXS
application. This is where you typically:

- choose or confirm the project directory
- select experimental and optional solvent SAXS data
- point the project at frames, PDB structures, and clusters folders
- choose a model template
- set the q-range, grid behavior, recognized elements, and excluded elements
- create or load computed distributions
- optionally compute project-backed representative structures or
  Debye-Waller factors
- build SAXS components with the selected component-build mode
- preview component traces and prior histograms before moving to Prefit

## Typical setup order

1. Process the MD trajectory first: inspect it, export frames, optionally
   convert XYZ frames to PDB, and extract the cluster folder you want to model.
2. Create a dedicated project folder for this SAXSShell session.
3. Open the SAXS application from the repository root:
   `PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs`.
4. Create a new project or load the project directory you prepared.
5. Select the experimental dataset and the cluster folder you want to model.
6. Choose the template, q-range, grid mode, and excluded elements.
7. Pick the **Component build mode** for the current modeling branch.
8. Click **Create Computed Distribution**.
9. Optionally compute representative structures from the full UI or the beta
   CLI setup if later workflows should use representative files instead of
   average cluster folders.
10. Optionally click **Compute Debye-Waller Factors (beta)** if the active
    clusters are PDB files and you want saved disorder terms for later
    workflows.
11. Click **Build SAXS Components**.
12. Review the component preview, cluster table, and prior histogram preview.
13. Move to **SAXS Prefit** once the active distribution has the component
    traces you want to fit.

!!! info "Image placeholder"
Add a screenshot focused on the project and input-selection controls used
in steps 2 through 4.

!!! info "Image placeholder"
Add a screenshot focused on the computed-distribution and component-build
controls used in steps 5 through 8.

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
with `No Contrast (Debye)`, one built with
`1D Born Approximation (Average)`, and one built with
`3D FFT Born Approximation`.

## Component build modes

The **Component build mode** dropdown controls what happens when you click
**Build SAXS Components**.

| Mode                                | What happens                                                                                                                                                                                                                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **No Contrast (Debye)**             | The main SAXS UI runs the direct Debye component builder and saves the component traces into the active computed distribution.                                                                                                                                                                                            |
| **Contrast (Debye)**                | SAXSShell opens the linked **SAXS Contrast Mode** workflow so you can analyze representative structures, compute electron-density terms, and build contrast-aware Debye traces for that computed distribution.                                                                                                            |
| **1D Born Approximation (Average)** | SAXSShell opens the linked legacy radial-density workflow in computed-distribution mode so you can compute per-stoichiometry electron-density profiles, apply optional solvent subtraction, evaluate spherical Fourier transforms, and then push the resulting Born-approximation components back into the model.         |
| **3D FFT Born Approximation**       | SAXSShell opens the separate Cartesian FFT workflow so you can build a 3D electron-density map, optionally apply a constant solvent-density contrast subtraction in real space, compare the q-shell-averaged FFT result against 1D Born and Debye references, and push computed traces back into the linked distribution. |

## Representative structures

Representative structures are optional project-backed files that compatible
Debye, Born, FFT, and RMCSetup workflows can use instead of average cluster
folders. Use **Tools > Structure Analysis > Open Representative Structures** for
the full interactive analysis UI, or use **Tools > (beta) > Open Representative
CLI Setup (Beta)** to save `representative_structure_cli_run.json` and run the
same backend from the source checkout:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.representativefinder run /path/to/project
```

## Debye-Waller factors

**Compute Debye-Waller Factors (beta)** is an optional linked step in Project
Setup.

!!! warning "Current testing status"
The linked **Compute Debye-Waller Factors (beta)** workflow is currently in
testing and has a known bug. Use it cautiously and verify any saved
Debye-Waller outputs before treating them as reliable downstream inputs.

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

!!! info "Image placeholder"
Add a screenshot of the Debye-Waller readiness indicator and button state
inside **Project Setup**, including an example tooltip if available.

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
- Finish the basic project definition in **Project Setup** before you spend
  time interpreting Prefit or DREAM behavior.
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
