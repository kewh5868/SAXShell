Store reference molecule PDB files for the xyz2pdb workflow in this folder.

Each reference should be a single-molecule PDB whose atom names are stable,
because the residue-assignment JSON anchors refer to those atom names.

Optional `.json` files with the same base name may also live beside each PDB.
These sidecars store preferred backbone atom pairs that the native mapper uses
to narrow the initial backbone search before full atom assignment.

You can populate this folder manually with PDB files, or use:

  xyz2pdb references add SOURCE_FILE --name REFERENCE_NAME

from the command line, or the "Add Reference Molecule" controls in the UI.
