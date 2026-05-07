# Representative Structure CLI (beta)

The representative structure CLI is a headless alternative to the full
Representative Structures UI. It uses the same representative-selection
workflow and writes the same project registry under
`rmcsetup/representative_structures/`, but avoids plot drawing, structure
viewer updates, and Qt progress refreshes during the actual analysis.

## Workflow

1. Open the main SAXSShell application from the source checkout.
2. Open **Tools > (beta) > Open Representative CLI Setup (Beta)**.
3. Select the project folder and representative input folder.
4. Load or enter the bond-pair and angle-triplet definitions.
5. Save the run file.
6. Run the printed command from the repository root in the target environment:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.representativefinder run /path/to/project
```

The default run file is saved at:

```text
/path/to/project/representative_structure_cli_run.json
```

## Run File

The beta setup window stores paths relative to the project folder when
possible. A typical run file includes:

```json
{
  "version": 1,
  "input_dir": "clusters_splitxyz0001",
  "output_dir": "representative_finder/representativefinder_batch_clusters_splitxyz0001",
  "analysis_mode": "all",
  "overwrite_existing": false,
  "settings": {
    "selection_algorithm": "target_distribution_quantile_distance",
    "bond_weight": 1.0,
    "angle_weight": 1.0,
    "solvent_weight": 1.0,
    "parallel_workers": 8
  }
}
```

`analysis_mode` can be `all` or `single`. In `single` mode, the run file can
also include `selected_stoichiometry`.

## CLI Commands

Inspect the targets without running:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.representativefinder inspect /path/to/project
```

Run the saved setup:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.representativefinder run /path/to/project
```

Use a non-default run file:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.representativefinder run /path/to/project --run-file /path/to/run.json
```

Override the saved worker count:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.representativefinder run /path/to/project --workers 12
```

Recompute bins that already have project representatives:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 python -m saxshell.representativefinder run /path/to/project --overwrite-existing
```

## Outputs

Each CLI target writes the same analysis artifacts as the full UI:

- `representative_selection.json`
- `candidate_scores.tsv`
- `selection_summary.txt`
- the copied representative structure

After each successful target, the CLI calls the same project-persistence path
as the UI. The project registry and reusable representative files are written
under:

```text
/path/to/project/rmcsetup/representative_structures/
```

For equivalent inputs, the full UI and CLI path should leave compatible
project metadata and representative structure sets.
