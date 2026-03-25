# Project Configuration

SAXS projects are file-backed. The application stores enough metadata in the
project directory to let you reopen a project and continue without rebuilding
everything from scratch each time.

## What a project captures

From the current project-manager and Prefit workflow code, a SAXS project can
persist:

- the selected experimental dataset
- the cluster directory used to build components
- the active model template
- prior-weight metadata
- SAXS component maps
- saved Prefit state
- saved DREAM settings and run artifacts
- cluster geometry metadata when the active template needs it

## Why this matters

Several behaviors in the UI depend on previously computed project state:

- Prefit can reuse saved cluster-geometry metadata without recomputing it every
  session.
- Deprecated templates can still be resolved when an older project names them,
  even though they are hidden by default in new template pickers.
- DREAM runtime bundles can be rebuilt from the current saved Prefit workflow
  state.

## Template-aware behavior

Template selection is part of project configuration, not just a UI preference.
That means the active template can affect:

- which extra runtime inputs are required
- whether cluster geometry metadata is mandatory
- which shape approximations are allowed
- which geometry-derived parameters are generated

## Practical guidance

- Keep one project directory per modeling attempt or dataset family.
- If you significantly change cluster folders, regenerate the component and
  geometry metadata instead of assuming older saved state is still valid.
- When comparing templates, treat the saved Prefit state as template-specific.

## Related pages

- [Project Setup](../getting-started/project-setup.md)
- [SAXS Prefit](saxs-prefit.md)
- [Template System](template-system.md)
