# Project Configuration

SAXS projects are file-backed, but the current UI is centered on
**computed distributions** inside each project. A project can hold multiple
distributions that share the same dataset family while differing in template,
build mode, q-range, excluded elements, or structure-weighting choices.

The project directory is also where the main SAXS UI meets the supporting
applications: cluster folders, component metadata, electron-density or
contrast-mode artifacts, saved PDF calculations, and fit results all end up
tied back to the same project state.

## What a project captures

From the current project-manager and Prefit workflow code, a SAXS project can
persist:

- the selected experimental and optional solvent datasets
- the frames, PDB structures, and clusters directories referenced by the
  project
- the active model template and Project Setup controls
- saved computed distributions under the project's distribution storage
- prior-weight metadata and SAXS component maps for each distribution
- saved Prefit state and DREAM run artifacts tied to a specific distribution
- cluster geometry metadata when the active template needs it
- saved supporting-application outputs such as PDF calculations,
  Debye-Waller analyses, contrast-mode artifacts, and electron-density
  mapping state

## What defines a computed distribution

In Project Setup, **Create Computed Distribution** saves the active modeling
snapshot. The current UI treats the following settings as part of the
distribution identity:

- selected template
- component build mode
- cluster source folder
- q-range and grid mode
- excluded elements
- observed-only versus observed-plus-predicted structure weighting
- model-only mode

When experimental-grid mode is active, the experimental data source also feeds
into that saved identity. This is why two distributions can coexist in one
project even when they differ only by build mode, for example
`No Contrast (Debye)` versus `Born Approximation (Average)`.

## Why this matters

Several behaviors in the UI depend on previously computed distribution state:

- the Project Setup dropdown can reload earlier computed distributions without
  overwriting the current one
- Prefit and DREAM reuse the component maps, prior weights, and saved runtime
  state from the active distribution
- supporting tools launched from **Build SAXS Components** inherit the active
  distribution context, not just the top-level project directory
- push-to-model actions in linked tools are intentionally locked once that
  distribution already has saved Prefit snapshots or DREAM runs

## Template-aware behavior

Template selection is part of project configuration, not just a UI preference.
That means the active template can affect:

- which extra runtime inputs are required
- whether cluster geometry metadata is mandatory
- which shape approximations are allowed
- which geometry-derived parameters are generated
- which computed distribution a later Prefit or DREAM run belongs to

## Practical guidance

- Keep one project directory per modeling attempt or dataset family.
- Treat computed distributions as intentional branches of that project rather
  than as temporary UI state.
- If you significantly change cluster folders, build mode, q-range, or
  excluded elements, create a fresh computed distribution instead of assuming
  older saved state is still valid.
- When comparing templates, treat Prefit and DREAM state as
  distribution-specific, not just project-specific.

## Related pages

- [Project Setup](../getting-started/project-setup.md)
- [GUI Overview](gui-overview.md)
- [Cluster Extraction](cluster-extraction.md)
- [SAXS Prefit](saxs-prefit.md)
- [pyDREAM Workflow](pydream-workflow.md)
- [PDF Calculation](pdf-calculation.md)
- [Template System](template-system.md)
