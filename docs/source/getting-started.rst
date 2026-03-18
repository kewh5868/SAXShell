:tocdepth: 2

.. index:: getting-started

.. _getting-started:

===============
Getting started
===============

``saxshell`` currently ships three application-oriented workflows that
fit together well for simulation-driven structure analysis:

1. ``mdtrajectory`` for inspecting a trajectory, choosing an
   equilibration cutoff, and exporting single-frame structures.
2. ``xyz2pdb`` for converting XYZ frames into residue-labeled PDB files
   using reference molecules plus residue-assignment rules.
3. ``clusters`` for cluster-network analysis on folders of extracted
   single-frame ``.pdb`` or ``.xyz`` files.

mdtrajectory
------------

Use ``mdtrajectory`` when you are starting from a multi-frame
trajectory file and need to inspect it, optionally choose a CP2K-based
steady-state cutoff, and export a clean folder of single-frame
structures.

Launch the UI ::

    mdtrajectory

Preview or export from the terminal ::

    mdtrajectory preview traj.xyz --use-cutoff --cutoff-fs 50
    mdtrajectory export traj.xyz --energy-file traj.ener \
        --use-suggested-cutoff --temp-target-k 300

Notebook usage ::

    from saxshell.mdtrajectory import MDTrajectoryWorkflow

    workflow = MDTrajectoryWorkflow("traj.xyz", energy_file="traj.ener")
    summary = workflow.inspect()
    preview = workflow.preview_selection(use_cutoff=True)
    result = workflow.export_frames(use_cutoff=True)

xyz2pdb
-------

Use ``xyz2pdb`` when you have one XYZ file or a folder of XYZ frames
and want to assign molecule names, residue names, and atom names before
writing PDB output.

The workflow relies on two inputs:

1. A reference-library folder of PDB files.
2. A JSON residue-assignment file that names each reference molecule,
   defines one or more anchor pairs, and optionally assigns residue
   names to remaining free atoms.

Launch the UI ::

    xyz2pdb

The UI can refresh the current reference-library folder into a dropdown
list and create a new reference PDB from a source ``.pdb`` or ``.xyz``
file.

UI use case ::

    xyz2pdb ui splitxyz \
        --config dmf_assignments.json \
        --library-dir src/saxshell/xyz2pdb/reference_library

Inside the UI:

1. Select the XYZ file or folder to convert.
2. Select the residue-assignment JSON file.
3. Confirm that the reference-library folder contains the expected
   molecule, such as ``dmf``.
4. Use ``Inspect`` to confirm the input and available references.
5. Use ``Preview`` to review the first-frame assignments.
6. Choose an output directory and run ``Convert XYZ to PDB``.

Reference-library helpers ::

    xyz2pdb references list
    xyz2pdb references add ref.xyz --name dmf --residue-name DMF

Example residue-assignment JSON ::

    {
      "molecules": [
        {
          "name": "PBI",
          "reference": "pbi",
          "residue_name": "PBI",
          "anchors": [
            {"pair": ["PB1", "I1"], "tol": 0.25}
          ]
        }
      ],
      "free_atoms": {
        "O": {"residue_name": "SOL", "atom_name": "O1"}
      }
    }

Preview or export from the terminal ::

    xyz2pdb preview splitxyz --config assignments.json \
        --library-dir references
    xyz2pdb export splitxyz --config assignments.json \
        --library-dir references

CLI use case ::

    xyz2pdb inspect splitxyz \
        --config dmf_assignments.json \
        --library-dir src/saxshell/xyz2pdb/reference_library
    xyz2pdb preview splitxyz \
        --config dmf_assignments.json \
        --library-dir src/saxshell/xyz2pdb/reference_library
    xyz2pdb export splitxyz \
        --config dmf_assignments.json \
        --library-dir src/saxshell/xyz2pdb/reference_library \
        --output-dir xyz2pdb_splitxyz

Notebook usage ::

    from saxshell.xyz2pdb import XYZToPDBWorkflow

    workflow = XYZToPDBWorkflow(
        "splitxyz",
        config_file="assignments.json",
        reference_library_dir="references",
    )
    inspection = workflow.inspect()
    preview = workflow.preview_conversion()
    result = workflow.export_pdbs()

Class-based use case ::

    from saxshell.xyz2pdb import XYZToPDBWorkflow

    workflow = XYZToPDBWorkflow(
        "splitxyz",
        config_file="dmf_assignments.json",
        reference_library_dir="src/saxshell/xyz2pdb/reference_library",
        output_dir="xyz2pdb_splitxyz",
    )
    inspection = workflow.inspect()
    preview = workflow.preview_conversion()
    result = workflow.export_pdbs()

clusters
--------

Use ``clusters`` when you want to analyze already extracted frame
folders as cluster networks. If residue-labeled PDB output matters for
your downstream analysis, ``xyz2pdb`` fits naturally between
``mdtrajectory`` and ``clusters``.

Typical progression:

1. Export frames with ``mdtrajectory``.
2. Run ``xyz2pdb`` if you want residue-aware PDB output.
3. Run ``clusters`` on the resulting PDB frames, or directly on XYZ
   frames if residue labels are not needed.

Launch the UI ::

    clusters

Preview or export from the terminal ::

    clusters preview splitxyz0001
    clusters export splitxyz0001 --node Pb --linker I --shell O \
        --pair-cutoff Pb:I:3.36 --pair-cutoff Pb:O:3.36

Notebook usage ::

    from saxshell.cluster import ClusterWorkflow

    workflow = ClusterWorkflow(
        "splitxyz0001",
        atom_type_definitions={"node": [("Pb", None)]},
        pair_cutoff_definitions={("Pb", "I"): {0: 3.36}},
    )
    summary = workflow.inspect()
    selection = workflow.preview_selection()
    result = workflow.export_clusters()
