Store reference molecule PDB files for the xyz2pdb workflow in this folder.

Each reference should be a single-molecule PDB whose atom names are stable,
because the residue-assignment JSON anchors refer to those atom names.

You can populate this folder manually with PDB files, or use:

  xyz2pdb references add SOURCE_FILE --name REFERENCE_NAME

from the command line, or the "Add Reference Molecule" controls in the UI.
