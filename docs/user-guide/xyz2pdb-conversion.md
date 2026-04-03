# XYZ to PDB Conversion

The `xyz2pdb` tool converts one `XYZ` file or a folder of `XYZ` frames into
molecule-aware `PDB` files using a reference library and interactive mapping
definitions. It is available as the standalone `xyz2pdb` application and from
the main SAXSShell window through `Tools > Open XYZ -> PDB Conversion`.

## Purpose

Use `xyz2pdb` when downstream steps need residue identity rather than only raw
atomic coordinates. Typical reasons include:

- preparing molecule-aware `PDB` frames for cluster extraction
- separating free ions or solvent atoms from mapped molecules
- reusing a consistent frame-to-frame atom-order template across a trajectory
- checking whether simulated molecules remain close to their reference geometry
- updating reference molecules from assertion-validated simulation averages

## Main capabilities

The current interface can:

- load either a single `XYZ` file or a folder of `XYZ` files
- analyze a sample frame and detect the element inventory automatically
- browse a reference-molecule library and create new references from `XYZ` or `PDB`
- define free atoms and reference-molecule mappings directly in the UI
- auto-fill mapped residue names from the selected reference entry
- edit direct bond tolerances as percentages rather than flat angstrom cutoffs
- show tight and relaxed min/max bond-search windows for every direct bond
- estimate molecule counts from frame stoichiometry before conversion
- keep hydrogen omission disabled by default and only test deprotonation when requested
- map the first frame in the background, then reuse that atom-order template for later frames
- show live progress and console output during conversion instead of blocking the whole UI
- optionally run assertion mode to write per-molecule files and compare internal distance distributions
- offer one-at-a-time reference updates or datetime-stamped reference versions for assertion-passing residues
- register the converted `PDB structure folder` back into the active SAXS project automatically

## Launching the tool

### From the main SAXSShell UI

Open the tool from:

- `Tools > Open XYZ -> PDB Conversion`
- the `Open XYZ -> PDB Conversion` button in the project setup workflow

When opened from an active SAXS project, the window is linked to that project.
The current project's `Frames folder` is used to prefill the input when
available, and successful exports write the output folder back to the project's
`PDB structure folder`.

### From the terminal

Installed package:

```bash
xyz2pdb
xyz2pdb ui path/to/frame_folder
```

From a source checkout:

```bash
PYTHONPATH=src conda run --no-capture-output -n saxshell-py312 \
  python -m saxshell.xyz2pdb.cli
```

## Typical workflow

1. Open `xyz2pdb`.
2. Choose an `XYZ` file or a folder of `XYZ` files.
3. Click `Analyze Input`.
4. Check the sample analysis and confirm the detected elements.
5. Review or create reference molecules in the library.
6. Add any free atoms that should remain standalone in the exported `PDB`.
7. Add each reference molecule you want to map and review its bond tolerances.
8. Click `Estimate PDB Mapping` and choose the estimate solution if more than one exists.
9. Optionally enable `Assertion Mode`.
10. Click `Convert XYZ Frames to PDB`.
11. If assertion mode passes for a residue type, review the preview dialog and decide whether to skip, replace the current reference, or save a new datetime-stamped reference version.

## Interface guide

The left side of the window is split into three resizable sections:

- `XYZ Input`
- `Reference Molecules`
- `PDB Mapping Definitions`

The right side contains the conversion controls, mapping summary, progress bar,
and live run log.

### XYZ Input

This section controls the source geometry.

Fields and controls:

- `XYZ input`
  Choose either one `.xyz` file or a folder that contains many `.xyz` frames.
- `Browse File`
  Select a single `XYZ` file.
- `Browse Folder`
  Select a directory of `XYZ` frames.
- `Analyze Input`
  Reads the first frame, detects whether the input is a single file or a frame
  folder, counts the elements, and populates the free-atom element list.
- `Sample Analysis`
  Shows the analyzed input path, input mode, number of `XYZ` files found,
  sample frame name, sample comment line, sample atom count, reference-library
  location, available references, and element counts.

Notes:

- The UI no longer requires a legacy JSON config file.
- The mapping definition now lives entirely in the native Qt interface.

### Reference Molecules

This section is split between reference creation and reference browsing.

#### Reference library controls

- `Reference library`
  Folder containing single-molecule reference `PDB` files.
- `Browse`
  Choose a different reference library folder.

The browser on the right shows:

- `Available references`
  Dropdown of the discovered reference molecules.
- `Refresh`
  Reloads the library folder.
- `Reference information`
  Displays the reference name, file path, residue name, atom count, preferred
  backbone pairs, and a preview of the atom names.

#### Add Reference Molecule

Use this when a needed reference does not already exist in the library.

Fields:

- `Source PDB/XYZ`
  Input structure file used to create the new reference.
- `Reference name`
  Filename stem for the saved reference.
- `Residue name`
  Optional residue code written into the saved reference. It must be exactly
  three capital letters.
- `Create Reference`
  Saves the new single-molecule reference into the current library folder.

Reference behavior:

- New references can be created from either `PDB` or `XYZ`.
- Atom names are normalized and stabilized during creation.
- A sidecar `.json` file is written beside the `PDB` to store preferred
  backbone pairs for faster matching.
- Assertion-derived updates can later overwrite a reference or create a new
  version with a timestamp in the name.

### PDB Mapping Definitions

This section defines how the analyzed `XYZ` atoms should be interpreted.

#### Free Atoms

Use this table for atoms that should not be absorbed into a larger reference
molecule.

Fields:

- `Element`
  Element selected from the sample frame's detected elements.
- `Residue`
  Three-letter residue code for the free atom in the exported `PDB`.
- `Add Free Atom`
  Adds the current element and residue to the table.
- `Remove Selected`
  Removes the selected free-atom rule.

Table columns:

- `Element`
- `Residue`

Rules:

- Each free-atom element can only be listed once.
- Residue codes must be exactly three capital letters.

#### Reference Molecules

Use this table to define every molecule type that should be matched from the
frame.

Controls:

- `Reference`
  Reference-library entry to match.
- `Residue`
  Residue name written into the exported `PDB`. This is auto-filled from the
  selected reference, but can be overridden with another three-letter code.
- `Missing H`
  Maximum number of reference hydrogens that may be omitted after both
  full-hydrogen passes fail.
- `Tight`
  Percentage multiplier applied to each bond's base tolerance percentage during
  the first matching pass.
- `Relaxed`
  Fallback percentage multiplier used only after the tight full-hydrogen pass
  fails.
- `Add Molecule`
  Appends the current mapping definition.
- `Update Selected`
  Replaces the currently selected molecule row with the edited values.
- `Remove Selected`
  Removes the selected molecule row.

Table columns:

- `Reference`
- `Residue`
- `Bonds`
- `Tight %`
- `Relaxed %`
- `Missing H`

Important behavior:

- `Missing H` defaults to `0`, so the program does not assume deprotonation on
  the first pass.
- The tool first tries full-hydrogen matching with the tight pass, then the
  relaxed pass, and only then tests hydrogen-omitted variants if you have
  allowed them.
- Relaxed full-hydrogen matches are treated as tolerance or geometry issues,
  not as missing-hydrogen matches.

#### Direct Bond Tolerances

This table shows and edits the direct bond windows used for matching the
currently selected reference molecule.

Columns:

- `Atom 1`
- `Atom 2`
- `Ref (A)`
- `Tolerance (%)`
- `Tight Min (A)`
- `Tight Max (A)`
- `Relaxed Min (A)`
- `Relaxed Max (A)`

How it works:

- `Tolerance (%)` is stored per bond, not as one global angstrom cutoff.
- The percentage is multiplied by that bond's reference length to generate the
  bond's absolute tolerance.
- `Tight` and `Relaxed` then scale that per-bond tolerance again for the two
  search passes.
- The min/max columns let you see the exact search window that each bond will
  use.

#### Hydrogen Handling

This section controls what happens if a molecule can only be matched after one
or more reference hydrogens are missing.

Modes:

- `Leave unassigned (Recommended)`
  Leave unmatched hydrogen atoms as free or unassigned atoms.
- `Assign orphaned hydrogen`
  Reassign a nearby unmatched hydrogen to a deprotonated site.
- `Restore missing hydrogen`
  Place the hydrogen at the reference-aligned position.

## Convert panel

The right-hand panel controls the actual export.

### Output directory

- `Output directory`
  Destination folder for the converted `PDB` frames.
- `Browse`
  Choose the output folder manually.

If you do not change it, the tool suggests a sibling output folder based on the
input path.

### Mapping Summary

This area shows:

- input analysis results after `Analyze Input`
- stoichiometric estimate details after `Estimate PDB Mapping`
- first-frame mapping results after conversion begins

### Estimate and solution selection

- `Estimate PDB Mapping`
  Solves the sample-frame stoichiometry using the current free-atom and
  reference definitions.
- `Estimate solution`
  Dropdown used when more than one complete stoichiometric solution is found.

Conversion reuses the current estimate if one is already available.

### Assertion Mode

`Assertion Mode` is off by default.

When enabled, `xyz2pdb` will:

- write individual molecule `PDB` files into an `assertion_molecules` folder
- compare each mapped molecule's internal pairwise distance distribution against
  the reference
- compare molecules of the same residue type against the rest of the exported set
- write an `assertion_report.txt`
- report per-residue median and max distribution drift in the log
- identify only the residue types that passed assertion and prepare averaged
  reference-update candidates for them

### Convert XYZ Frames to PDB

This launches the conversion in a background worker so the rest of the main
SAXSShell UI can stay responsive.

Runtime behavior:

- the first frame is mapped with the full molecule search
- progress messages report backbone searches, matched counts, and file writing
- later frames reuse the first frame's atom-order template when the atom order
  is unchanged
- conversion runs at low thread priority rather than monopolizing the whole UI

### Progress and run log

- `Progress`
  Shows step-based progress such as estimate reuse, first-frame mapping,
  per-file writes, and assertion analysis.
- `Run Log`
  Streams matching diagnostics, warnings, backbone-pair counts, template reuse
  messages, assertion summaries, and any project-folder registration messages.

## Future Search Mode Note

This is not implemented yet, but the current design backlog includes an
optional alternate search mode for `Estimate PDB Mapping`, test/preview
mapping, and full conversion.

Concept:

- represent each reference atom with a `CPK`-style volume rather than treating
  atoms only as points
- fix a candidate backbone pair first, then rotate the reference around that
  backbone axis
- score the placement by same-element overlap percent between the rotated
  reference volume and the local `XYZ` atom neighborhood
- map `tight` and `relaxed` settings onto stricter or looser overlap-acceptance
  thresholds
- use the overlap score as a screening stage before the final atom assignment
  and bond validation

Why preserve the idea:

- it may offer a more intuitive way to score axial rotations around a fixed
  backbone
- it could help reject obviously poor fits earlier in dense systems
- it would provide a second search strategy for cases where the current
  point-based matcher is difficult to tune

TODO:

- add an optional alternate search mode to the UI for estimate/test/convert
- document how `tight` and `relaxed` settings translate into overlap thresholds
- compare overlap-scored screening against the current point-based search on
  representative systems before enabling it by default

## Assertion-derived reference updates

At the end of conversion, if assertion mode produced residue types that passed,
the tool presents them one at a time in a confirmation dialog.

Each dialog shows:

- the current reference structure
- the averaged structure derived from the passed molecules
- a small native ball-and-stick preview for both
- molecule count and assertion spread metrics
- the proposed datetime-stamped version name

Actions:

- `Skip`
  Ignore this candidate and move to the next one.
- `Save New Version`
  Save the averaged structure as a new reference such as
  `dmso_20260401_153012.pdb`.
- `Replace Existing Reference`
  Overwrite the current reference file with the averaged structure.

Only residues that passed assertion are offered in this flow.

## Output files

The main conversion writes:

- one `PDB` file per `XYZ` frame in the selected output directory

If assertion mode is enabled, it also writes:

- `assertion_molecules/RES/FRAME__RES_NNNN.pdb` per matched molecule
- `assertion_molecules/assertion_report.txt`
- `assertion_molecules/reference_update_candidates/*.pdb` averaged candidate references for passed residue types

## Reference library notes

The bundled reference library stores:

- one single-molecule `PDB` per reference
- an optional same-name `.json` sidecar that records preferred backbone pairs

Bundled references currently include:

- `dmso`
- `dmf`
- `ma`

The mapper uses preferred backbone pairs first, then falls back to the broader
anchor search if needed.

## CLI note

The standalone `xyz2pdb` command still exposes older CLI subcommands such as
`inspect`, `preview`, and `export`, including JSON-driven workflows for
scripting. The Qt interface documented here is the newer native mapping UI and
does not require the legacy JSON input file.

## Related pages

- [GUI Overview](gui-overview.md)
- [Cluster Extraction](cluster-extraction.md)
- [Results and Export](results-and-export.md)
- [Blender Structure Renderer](blender-structure-renderer.md)
