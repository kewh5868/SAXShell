# Bond Analysis

The **Bond Analysis** tool is SAXSShell's structure-analysis application for
measuring bond-length and angle distributions from stoichiometry-sorted cluster
folders.

## Launching the application

Open the tool from the main SAXS UI through
`Tools > Structure Analysis > Open Bond Analysis`.

When the window is launched from an active SAXS project, the current project
and cluster-folder reference are carried into the tool. If you change the
selected clusters folder there, that reference is saved back to the project.

## What the tool does

The current UI supports:

- choosing one sorted clusters directory as the analysis source
- saving results into a separate output directory
- limiting the run to checked stoichiometry labels
- defining bond-pair cutoffs directly in a table
- defining angle triplets directly in a table
- loading built-in presets and saving custom presets for later reuse
- reopening an existing bond-analysis output folder and browsing its saved
  distributions

The right side of the window focuses on the computed distributions. You can
refresh a results directory, select one or more saved bond-pair or angle
entries, and open them in a dedicated plot window. Matching items from multiple
cluster types can be overlaid together for comparison.

## Typical workflow

1. Start from the project's sorted clusters folder.
2. Confirm or choose the bond-analysis output directory.
3. Refresh the detected cluster types and clear any stoichiometries you do not
   want to include.
4. Load a preset or define the bond pairs and angle triplets manually.
5. Run the calculation and inspect the saved distributions from the results
   browser.

## Related pages

- [MD Extraction and Cluster Preparation](cluster-extraction.md)
- [Debye-Waller Analysis](debye-waller-analysis.md)
- [GUI Overview](gui-overview.md)
