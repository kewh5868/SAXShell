# API Overview

This section is intentionally lightweight. The repository does not yet include a
fully automated API reference pipeline, so this page focuses on the workflow
classes that are most likely to be imported directly.

## Recommended Python workflow classes

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

## Source-checkout launch note

Several tools can also be launched through their Python modules while the
public Python API stabilizes. From the repository root, start the main SAXS
application with:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.saxs
```

## TODO

TODO: expand this section if the repository adds a stable automatic API-docs
generation path later.
