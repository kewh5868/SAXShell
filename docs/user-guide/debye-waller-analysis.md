# Debye-Waller Analysis

The **Debye-Waller Analysis** tool is a separate supporting application for
estimating pair-resolved thermal-displacement coefficients from sorted PDB
cluster folders. It is designed to stay project-aware so the results can be
saved, reopened, and reused by later SAXSShell workflows.

## Launching the application

Open the tool from the main SAXS UI through
`Tools > Structure Analysis > Open Debye-Waller Analysis`.

If the tool is launched from an active project, it uses that project to:

- prefill the sorted clusters folder when a project cluster reference exists
- choose a project-aware default output location
- restore a previously saved Debye-Waller analysis when one already exists

## Input requirements

The workflow expects a sorted clusters folder whose subdirectories are the
stoichiometry bins to evaluate.

Important constraints:

- input structures must be `PDB` files
- `XYZ` cluster files are rejected
- contiguous-frame grouping is inferred from frame-numbered filenames inside
  each stoichiometry folder

The PDB requirement matters because the tool uses residue, sequence, and
segment metadata to separate intra-molecular and inter-molecular atom pairs.

## What the tool computes

For each stoichiometry label, the workflow:

1. detects the contiguous frame sets available in that stoichiometry folder
2. groups atoms into molecules from the PDB metadata
3. identifies intra-molecular and inter-molecular atom-pair types
4. computes Debye-Waller coefficients for each contiguous frame set
5. averages those segment-level values into stoichiometry-level summaries with
   preserved spread

It also builds an aggregated view across all stoichiometries, so one combined
coefficient row is available for each pair type and scope
(`intra-molecular` or `inter-molecular`) with the spread preserved.

## Live run feedback

The run log is intended to make the workflow easy to sanity-check while it is
running. It reports:

- validation and output-path setup
- how many contiguous frame sets were found for each stoichiometry label
- how many frames belong to each contiguous frame set
- per-segment progress as coefficient rows are generated
- final artifact locations

The tables update while the run is still active. The current tabs are:

- `Stoichiometries`
- `Aggregated Pairs`
- `Pair Types`
- `Scopes`
- `Segments`
- `Log`

The `Stoichiometries` tab is a lightweight sanity check on the parsed PDB
content. It shows metrics such as frame counts, contiguous-set counts, residue
names, element inventory, average atoms per frame, average molecule groups per
frame, and the most common molecule signatures.

## Project save and restore behavior

The tool includes **Save Current Analysis to Project** for manually storing the
current result in the active project.

It also auto-saves the first valid Debye-Waller analysis for a project when no
project-saved analysis already exists. When the tool is reopened from that same
active project, the saved analysis is loaded and the tables are repopulated
without rerunning the calculation.

## Saved outputs

Run artifacts are written to the selected output directory, and project-backed
copies are stored under the project's `exported_results/data/debye_waller/`
area.

The saved bundle includes:

- a summary JSON payload
- an aggregated pair-summary CSV across all stoichiometries
- a per-stoichiometry pair-summary CSV
- a per-stoichiometry scope-summary CSV
- a segment-level CSV with the contiguous-set rows

That keeps both the detailed per-stoichiometry data and the cross-stoichiometry
aggregate coefficients available for future applications.

## Related pages

- [MD Extraction and Cluster Preparation](cluster-extraction.md)
- [Bond Analysis](bond-analysis.md)
- [Results and Export](results-and-export.md)
- [GUI Overview](gui-overview.md)
