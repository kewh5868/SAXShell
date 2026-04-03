# SAXS Contrast Mode

This page is the contributor-facing map for the contrast-enabled SAXS component
workflow.

The design goal is simple: keep the contrast workflow fully separated from the
legacy no-contrast SAXS builder except at explicit routing and reload seams.

## Main entry points

The feature spans three layers:

- main SAXS UI routing
- contrast supporting-application workflow
- distribution-scoped persistence and reload

The main files are:

- `src/saxshell/saxs/contrast/settings.py`
- `src/saxshell/saxs/contrast/representatives.py`
- `src/saxshell/saxs/contrast/mesh.py`
- `src/saxshell/saxs/contrast/electron_density.py`
- `src/saxshell/saxs/contrast/debye.py`
- `src/saxshell/saxs/contrast/ui/main_window.py`
- `src/saxshell/saxs/contrast/ui/structure_viewer.py`
- `src/saxshell/saxs/project_manager/project.py`
- `src/saxshell/saxs/ui/project_setup_tab.py`
- `src/saxshell/saxs/ui/main_window.py`

## Workflow stages

The current contrast flow is staged as:

1. representative-structure analysis per stoichiometry bin
2. retained mesh construction and electron-density estimation
3. contrast-scaled Debye trace generation
4. distribution-scoped artifact persistence
5. automatic main-UI reload of the saved contrast distribution

The main UI routes into the contrast workflow when the active
`component_build_mode` is `contrast`. The no-contrast builder path remains the
default for `no_contrast`.

## Persistence model

Contrast mode uses the same computed-distribution system as the rest of the SAXS
workflow, but the distribution identity includes the build mode. The saved
distribution snapshot carries a `contrast/` subtree alongside the usual SAXS
component and prior files.

That subtree is expected to contain:

- `representative_structures/`
- `screening/`
- `geometry/`
- `electron_density/`
- `debye/`
- summary files copied at the contrast root for reopen convenience

The reopen path should read only from the saved distribution snapshot, not from
the transient working folder in `project/contrast_workflow`.

## Prior-weight behavior

No-contrast prior generation still maps stoichiometry bins to the original
cluster-bin representative files.

Contrast-mode prior generation now maps stoichiometry bins to the saved
representative-trace artifacts from the active contrast distribution. The bin
weights still come from the observed cluster counts, but the trace linkage comes
from the saved contrast summary.

## Current simplifications

These are intentional and should be treated as explicit constraints unless a
follow-up change is planned:

- one existing representative structure per stoichiometry bin
- no predicted-structure weighting in the contrast build path
- support-app UI currently defaults to the built-in neat-solvent estimate
  (`H2O`, `1.0 g/mL`) rather than exposing the full solvent-settings surface
- the structure viewer is a workflow/inspection view, not a publication renderer

## Test coverage

Focused coverage currently lives in:

- `tests/test_saxs_ui.py`
- `tests/test_saxs_contrast.py`

Those tests cover the mode selector, Tools-menu launch, representative analysis,
electron-density backends, mesh/density reload, distribution identity,
main-UI handoff, reopen behavior, and contrast-mode prior-weight mapping.
