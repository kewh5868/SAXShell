# SAXS Contrast Mode

The **SAXS Contrast Mode** workflow is the contrast-enabled alternative to the
standard no-contrast SAXS component builder. It lives in a separate supporting
application so the original Debye component-build path stays intact.

Use contrast mode when you want the SAXS component build to be tied to:

- one saved representative structure per stoichiometry bin
- a retained geometry mesh for each representative
- cluster and solvent electron-density estimates
- contrast-scaled representative traces that can be reopened later

## Where to open it

You can open the contrast workflow in two ways:

- **Project Setup > Component build mode > Contrast (Debye)**, then click
  **Build SAXS Components**
- **Tools > SAXS Calculation Preview > Open SAXS Contrast Mode**

When a saved contrast-mode distribution is active, Project Setup also exposes
**View Representative Structures** below **Build SAXS Components** so you can
reopen the saved representative-selection, mesh, density, and trace context.

## Typical workflow

1. In **Project Setup**, choose **Contrast (Debye)** above
   **Build SAXS Components**.
2. Open the contrast tool from **Build SAXS Components** or from **Tools**.
3. Confirm the inherited project folder, cluster folder, template, and q-range.
4. Run **Analyze Representative Structures** to select one existing structure
   per stoichiometry bin.
5. Run **Compute Electron Density** to generate the retained mesh plus cluster
   and solvent density terms.
6. Run **Build Contrast SAXS Components** to write the component traces and
   push the finished computed distribution back into the main SAXS UI.
7. Generate prior weights, continue in **SAXS Prefit**, and then continue into
   **SAXS DREAM Fit** as usual.

## What gets saved

Contrast-mode computed distributions retain the normal SAXS component outputs
plus a distribution-scoped `contrast/` artifact folder. In practice that saved
snapshot includes:

- copied representative structure files
- screening summaries and tables
- retained mesh geometry files
- electron-density summaries
- contrast Debye trace summaries

Those saved artifacts are what power the reopen flow from
**View Representative Structures** and from the Tools entry when a contrast-mode
distribution is active.

## Computed-distribution behavior

The SAXS component build mode is part of computed-distribution identity. That
means two distributions can coexist even if everything else matches, as long as
one is **No Contrast (Debye)** and the other is **Contrast (Debye)**.

Within contrast mode:

- prior weights still use stoichiometry-bin counts and weights
- the prior entries point to the saved representative-trace files for that
  distribution
- Prefit and DREAM load the same saved contrast-mode distribution once it has
  been pushed back into the main UI

## Current assumptions and limitations

The current implementation intentionally starts with a narrower scope:

- one existing representative structure is selected per stoichiometry bin
- the support app currently uses the built-in neat-solvent default
  `H2O, 1.0 g/mL` from the UI path unless you run the lower-level backend
  directly
- contrast mode currently works on the observed cluster bins and does not yet
  combine predicted-structure weights into the contrast build path
- the retained mesh is designed as a reusable workflow artifact and inline
  visualizer overlay, not as a publication-grade renderer

These are workflow simplifications, not statements about the underlying theory.

## Related pages

- [GUI Overview](gui-overview.md)
- [Project Configuration](project-configuration.md)
- [SAXS Prefit](saxs-prefit.md)
- [pyDREAM Workflow](pydream-workflow.md)
