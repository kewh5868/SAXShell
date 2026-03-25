# Results and Export

SAXSShell stores more than one kind of output. Some files are meant for people,
and some are meant for the workflow itself.

## Prefit outputs

Prefit can currently save or export:

- saved Prefit state snapshots
- plot data exports
- best-prefit presets
- cluster geometry metadata for geometry-aware templates

## DREAM outputs

The DREAM workflow produces several layers of output:

- the runtime bundle used to launch the run
- full sampler artifacts in the DREAM run directory
- condensed exports intended for easier inspection or handoff
- statistics and model-fit export bundles
- violin-data exports

## Why the distinction matters

When debugging or reproducing a run:

- use the full DREAM artifacts for the authoritative record
- use condensed exports for quick comparisons or sharing
- use the saved Prefit state to reconstruct the model that generated a bundle

## Downstream reuse

The repository also contains `fullrmc` tooling that consumes SAXS project
artifacts downstream. That makes careful project saving and export discipline
worthwhile, especially when a project will be reopened later.

## TODO

TODO: document the exact condensed-export file set and naming conventions once
the current export schema is declared stable.
