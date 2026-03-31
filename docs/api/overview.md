# API Overview

This section is intentionally lightweight. The repository does not yet include a
fully automated API reference pipeline, so this page focuses on the workflow
classes that are most likely to be imported directly.

## Recommended entry points

### Trajectory processing

- `saxshell.mdtrajectory.workflow.MDTrajectoryWorkflow`

### Cluster extraction

- `saxshell.cluster.workflow.ClusterWorkflow`

### Cluster dynamics

- `saxshell.clusterdynamics.workflow.ClusterDynamicsWorkflow`
- `saxshell.clusterdynamicsml.workflow.ClusterDynamicsMLWorkflow`

### Bond analysis

- `saxshell.bondanalysis.workflow.BondAnalysisWorkflow`

### PDF calculation

- `saxshell.pdf.debyer.workflow.DebyerPDFWorkflow`

### XYZ to PDB conversion

- `saxshell.xyz2pdb.workflow.XYZToPDBWorkflow`

### SAXS prefit workflow

- `saxshell.saxs.prefit.workflow.SAXSPrefitWorkflow`

## Template support packages

The SAXS stack also includes reusable modules for:

- template loading
- template installation
- Prefit cluster geometry metadata
- DREAM runtime bundle generation and result loading

These modules are usable from Python, but their interfaces are evolving faster
than the main workflow classes above.

## CLI-first note

Several parts of the repository are easier to discover from their CLI help than
from their Python surface:

```bash
saxshell --help
saxshell saxs --help
clusters --help
mdtrajectory --help
```

From a source checkout, you can inspect the SAXS CLI directly with:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs --help
```

## TODO

TODO: expand this section if the repository adds a stable automatic API-docs
generation path later.
