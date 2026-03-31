|Icon| |title|_
===============

.. |title| replace:: saxshell
.. _title: https://kewh5868.github.io/saxshell

.. |Icon| image:: https://raw.githubusercontent.com/kewh5868/SAXSShell/main/docs/source/img/saxshell_icon.png
        :target: https://kewh5868.github.io/saxshell
        :height: 100px

|PyPI| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/kewh5868/saxshell/actions/workflows/matrix-and-codecov-on-merge-to-main.yml/badge.svg
        :target: https://github.com/kewh5868/saxshell/actions/workflows/matrix-and-codecov-on-merge-to-main.yml

.. |Codecov| image:: https://codecov.io/gh/kewh5868/saxshell/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/kewh5868/saxshell

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/saxshell
        :target: https://anaconda.org/conda-forge/saxshell

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff
        :target: https://github.com/kewh5868/saxshell/pulls

.. |PyPI| image:: https://img.shields.io/pypi/v/saxshell
        :target: https://pypi.org/project/saxshell/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/saxshell
        :target: https://pypi.org/project/saxshell/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/kewh5868/saxshell/issues

Python toolkit for analyzing small-angle X-ray scattering from
molecular dynamics-derived liquid structures.

``saxshell`` is intended for researchers who want reproducible,
scriptable workflows that connect atomistic simulation output to
scattering observables and structural interpretation.

The project is being developed as a Python library with command-line
entry points and documentation for simulation-driven scattering
analysis, especially for liquid-state structure and solvation-focused
studies.

For more information about the saxshell library, please consult our `online documentation <https://kewh5868.github.io/saxshell>`_.

Citation
--------

If you use saxshell in a scientific publication, we would like you to cite this package as

        saxshell Package, https://github.com/kewh5868/saxshell

Installation
------------

The preferred method is to use `Miniconda Python
<https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_
and install from the "conda-forge" channel of Conda packages.

To add "conda-forge" to the conda channels, run the following in a terminal. ::

        conda config --add channels conda-forge

We want to install our packages in a suitable conda environment.
The following creates and activates a new environment named ``saxshell_env`` ::

        conda create -n saxshell_env saxshell
        conda activate saxshell_env

The output should print the latest version displayed on the badges above.

If the above does not work, you can use ``pip`` to download and install the latest release from
`Python Package Index <https://pypi.python.org>`_.
To install using ``pip`` into your ``saxshell_env`` environment, type ::

        pip install saxshell

If you prefer to install from sources, after installing the dependencies, obtain the source archive from
`GitHub <https://github.com/kewh5868/saxshell/>`_. Once installed, ``cd`` into your ``saxshell`` directory
and run the following ::

        pip install .

This package also provides command-line utilities. To check the software has been installed correctly, type ::

        saxshell --version

You can also type the following command to verify the installation. ::

        python -c "import saxshell; print(saxshell.__version__)"


To view the basic usage and available commands, type ::

        saxshell -h

Getting Started
---------------

You may consult our `online documentation <https://kewh5868.github.io/saxshell>`_ for tutorials and API references.

Workflow Process Tree
---------------------

The intended end-to-end ``saxshell`` workflow is:

::

        Input MD trajectory
        (currently CP2K-style position trajectories, most commonly ``.xyz``;
        CP2K-style ``.pdb`` trajectories are also supported)
        |
        +-- ``mdtrajectory``
        |   |- Inspect the trajectory and optional CP2K ``.ener`` file.
        |   |- Choose an equilibration cutoff from the energy plot or use the
        |   |  suggested steady-state cutoff.
        |   `- Export single-frame structures into a sibling folder beside the
        |      source trajectory, such as ``splitxyz/`` or ``splitxyz_t50fs/``.
        |
        +-- Optional: ``xyz2pdb``
        |   |- Input: exported XYZ frames.
        |   |- Use reference molecule files plus a residue-assignment JSON file
        |   |  to identify molecules and label residues/atoms in each frame.
        |   `- Output: a PDB-frame folder for residue-aware downstream analysis.
        |
        +-- ``clusters``
        |   |- Input: a folder of extracted XYZ or PDB frames.
        |   |- Build cluster-network outputs and sort extracted clusters by
        |   |  stoichiometry.
        |   |- Include or exclude solvent molecules according to the selected
        |   |  shell/export settings.
        |   |- Output: a sibling folder such as ``clusters_splitxyz0001/``.
        |   `- Resume support: partial progress is tracked in
        |      ``cluster_extraction_metadata.json`` so interrupted runs can be
        |      resumed with the same settings.
        |
        +-- ``bondanalysis``
        |   |- Input: the resulting ``clusters_*`` directory.
        |   |- Use the stoichiometry-sorted cluster files to measure bond-pair
        |   |  and angle distributions.
        |   `- Output: a sibling ``bondanalysis_*`` folder containing analysis
        |      tables, histograms, plots, and a JSON manifest.
        |
        `-- Downstream SAXS modeling
            |- Plot cluster histograms and distributions
            |  (additional tooling is still being developed).
            |- Build a SAXS prefit model from the analyzed cluster population.
            `- Run Bayesian optimization with the DREAM algorithm.

In practice, the main branch point is whether you need residue-aware PDB
frames before clustering. If not, you can export frames with
``mdtrajectory`` and move directly into ``clusters`` using XYZ input.
If you do need molecule identification and residue labels, insert
``xyz2pdb`` between ``mdtrajectory`` and ``clusters``.

The final SAXS modeling application is still under active development,
so the currently mature workflow is trajectory inspection and export,
optional XYZ-to-PDB conversion, cluster extraction, and bond-pair /
angle analysis.

Applications
------------

``saxshell`` includes application-oriented workflows for common
simulation analysis tasks.

``mdtrajectory``
~~~~~~~~~~~~~~~~

The ``mdtrajectory`` application is designed for inspecting molecular
dynamics trajectories, selecting an equilibration cutoff, previewing
the resulting frame selection, and exporting a new set of output
frames.

The current workflow supports CP2K-style ``.xyz`` and ``.pdb``
trajectory files, with optional CP2K ``.ener`` files for temperature-
based steady-state cutoff suggestions.

The same ``mdtrajectory`` workflow can be used in three ways:

1. As a Qt desktop application for interactive use.
2. As a terminal command for scripted or batch workflows.
3. As a Python class in notebooks and other Python scripts.

To launch the Qt application, use one of the following commands ::

        mdtrajectory
        mdtrajectory ui
        saxshell mdtrajectory
        python -m saxshell.mdtrajectory

Terminal Use Cases
------------------

The ``mdtrajectory`` command-line interface is useful when you want to
inspect a run quickly, automate frame exports, or include trajectory
preprocessing inside a larger shell workflow.

Inspect a trajectory and optional energy file ::

        mdtrajectory inspect traj.xyz --energy-file traj.ener

Suggest a steady-state cutoff from a CP2K energy profile ::

        mdtrajectory suggest-cutoff traj.xyz \
            --energy-file traj.ener \
            --temp-target-k 300 \
            --temp-tol-k 1.0 \
            --window 3

Preview the selected export range without writing files ::

        mdtrajectory preview traj.xyz \
            --use-cutoff \
            --cutoff-fs 50 \
            --start 0 \
            --stride 2

Export frames directly from the terminal ::

        mdtrajectory export traj.xyz \
            --energy-file traj.ener \
            --use-suggested-cutoff \
            --temp-target-k 300 \
            --window 3

The ``saxshell`` command also forwards to the same workflow ::

        saxshell mdtrajectory inspect traj.xyz

Python and Notebook Use
-----------------------

For notebook use, the ``MDTrajectoryWorkflow`` class exposes the same
inspection, cutoff-selection, preview, and export functionality used
by the UI and terminal interfaces.

::

        from saxshell.mdtrajectory import MDTrajectoryWorkflow

        workflow = MDTrajectoryWorkflow(
            "traj.xyz",
            energy_file="traj.ener",
        )
        summary = workflow.inspect()
        suggested = workflow.suggest_cutoff(
            temp_target_k=300.0,
            temp_tol_k=1.0,
            window=3,
        )
        workflow.set_selected_cutoff(suggested.cutoff_time_fs)
        preview = workflow.preview_selection(use_cutoff=True)
        result = workflow.export_frames(use_cutoff=True)

``xyz2pdb``
~~~~~~~~~~~

The ``xyz2pdb`` application converts one XYZ file or a folder of XYZ
files into residue-labeled PDB files. It uses:

1. A reference-molecule library of PDB files.
2. A JSON residue-assignment file that describes which reference
   molecules to find, which anchor atom pairs should be used to place
   them, and how remaining free atoms should be named.

The same ``xyz2pdb`` workflow can be used in three ways:

1. As a Qt desktop application for interactive use.
2. As a terminal command for scripted or batch workflows.
3. As a Python class in notebooks and other Python scripts.

To launch the Qt application, use one of the following commands ::

        xyz2pdb
        xyz2pdb ui
        saxshell xyz2pdb
        python -m saxshell.xyz2pdb

Reference Library
-----------------

``xyz2pdb`` reads reference molecule PDB files from a reference-library
folder. In a source checkout, the bundled folder lives at
``src/saxshell/xyz2pdb/reference_library/``. The UI refreshes that
folder into a dropdown list of available references, and it can also
create a new reference PDB from a source ``.pdb`` or ``.xyz`` file and
add it to the current library folder.

List the available reference molecules ::

        xyz2pdb references list

Create a new reference molecule from the terminal ::

        xyz2pdb references add ref.xyz --name dmf --residue-name DMF

Residue Assignment JSON
-----------------------

The residue-assignment file is a JSON document. Each molecule entry
references a PDB in the library, names the output residue, and defines
one or more anchor pairs using atom names from the reference PDB.

::

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

XYZ2PDB UI Use Case
-------------------

One common use case is converting a folder of extracted XYZ frames into
residue-labeled PDB files for downstream structure analysis.

Suppose you have:

1. A folder of XYZ frames called ``splitxyz/``.
2. A reference molecule PDB such as ``dmf.pdb`` in the reference
   library folder.
3. A residue-assignment JSON file called ``dmf_assignments.json``.

Launch the UI ::

        xyz2pdb

Or prefill the main inputs from the terminal ::

        xyz2pdb ui splitxyz \
            --config dmf_assignments.json \
            --library-dir src/saxshell/xyz2pdb/reference_library

Inside the window:

1. Select the XYZ input file or folder.
2. Select the residue-assignment JSON file.
3. Confirm the reference-library folder and inspect the dropdown list
   of available reference molecules such as ``dmf``.
4. If needed, create a new reference PDB from a source ``.pdb`` or
   ``.xyz`` file using the ``Add Reference Molecule`` section.
5. Click ``Inspect`` to confirm the XYZ input and loaded references.
6. Click ``Preview`` to review the first-frame molecule and residue
   assignments.
7. Choose the output directory and click ``Convert XYZ to PDB``.

CLI Use Case
------------

The same DMF conversion workflow can be run completely from the
terminal.

List the available references first if you want to confirm that the
expected PDB is in the library ::

        xyz2pdb references list \
            --library-dir src/saxshell/xyz2pdb/reference_library

Inspect the selected XYZ input and JSON config ::

        xyz2pdb inspect splitxyz \
            --config dmf_assignments.json \
            --library-dir src/saxshell/xyz2pdb/reference_library

Preview the first frame before writing files ::

        xyz2pdb preview splitxyz \
            --config dmf_assignments.json \
            --library-dir src/saxshell/xyz2pdb/reference_library

Export the converted PDB files ::

        xyz2pdb export splitxyz \
            --config dmf_assignments.json \
            --library-dir src/saxshell/xyz2pdb/reference_library \
            --output-dir xyz2pdb_splitxyz

Terminal Use Cases
------------------

Inspect the selected XYZ input, reference library, and config ::

        xyz2pdb inspect splitxyz \
            --config assignments.json \
            --library-dir references

Preview the first-frame residue assignments and suggested output folder ::

        xyz2pdb preview splitxyz \
            --config assignments.json \
            --library-dir references

Export PDB files from one XYZ or a folder of XYZ files ::

        xyz2pdb export splitxyz \
            --config assignments.json \
            --library-dir references

The ``saxshell`` command also forwards to the same workflow ::

        saxshell xyz2pdb inspect splitxyz --config assignments.json

Python and Class-Based Use Case
-------------------------------

The same conversion can also be driven directly from Python code, which
is useful for notebooks, scripted preprocessing, and regression tests.

::

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

        print(inspection.configured_reference_names)
        print(preview.molecule_counts)
        print(result.written_files[0])

Python and Notebook Use
-----------------------

For notebook use, the ``XYZToPDBWorkflow`` class exposes the same
inspection, preview, reference-library, and export behavior used by the
UI and terminal interfaces.

::

        from saxshell.xyz2pdb import XYZToPDBWorkflow

        workflow = XYZToPDBWorkflow(
            "splitxyz",
            config_file="assignments.json",
            reference_library_dir="references",
        )
        inspection = workflow.inspect()
        preview = workflow.preview_conversion()
        result = workflow.export_pdbs()

``clusters``
~~~~~~~~~~~~

The ``clusters`` application is designed for solute cluster-network
analysis on folders of already extracted single-frame ``.pdb`` or
``.xyz`` files. A typical workflow is:

1. Use ``mdtrajectory`` to export a frames folder.
2. Use ``xyz2pdb`` when you want residue-labeled PDB frames generated
   from reference molecules and atom/residue assignments.
3. Open ``clusters`` on the resulting PDB frames, or skip ``xyz2pdb``
   and open ``clusters`` directly on XYZ frames when residue labels are
   not needed.
4. Confirm the detected mode (PDB or XYZ), estimated box dimensions,
   and whether periodic boundary conditions should be used.
5. Adjust the node, linker, shell, and pair-cutoff definitions.
6. Preview the output folder and export the cluster files.

The Qt UI, terminal commands, and ``ClusterWorkflow`` notebook API all
use the same cluster-analysis backend, so a preview or export run
should behave the same way across those interfaces.

Each cluster export also writes a
``cluster_extraction_metadata.json`` file into the output folder. That
metadata captures the extraction parameters, progress state, and
completed frames so you can review previous settings or resume an
interrupted run by exporting again with the same conditions into the
same folder.

The same ``clusters`` workflow can be used in three ways:

1. As a Qt desktop application for interactive use.
2. As a terminal command for scripted or batch workflows.
3. As a Python class in notebooks and other Python scripts.

To launch the Qt application, use one of the following commands ::

        clusters
        clusters ui
        saxshell cluster
        python -m saxshell.clusters

Cluster UI Use
--------------

The Qt interface is the easiest way to inspect an extracted frames
folder, review the estimated box dimensions, and decide whether to
enable periodic boundary conditions before exporting clusters.

Launch the UI ::

        clusters

Or open the UI with a frames folder preloaded ::

        clusters ui splitxyz0001

Inside the window:

1. Select the extracted frames folder produced by ``mdtrajectory``.
2. Click ``Inspect Frames Folder`` to confirm the detected mode.
3. Review the ``Selection Preview`` for the estimated box dimensions.
4. Leave box dimensions on ``Auto`` to use those estimates when PBC is
   enabled, or enter manual values.
5. Turn on ``Use periodic boundary conditions`` only when you want
   minimum-image clustering.
6. Edit the example ``Pb``, ``I``, and ``O`` rules to match your
   chemistry, then export the clusters.

Cluster Terminal Use
--------------------

The ``clusters`` command supports inspect, preview, and export
operations directly from the terminal.

Inspect an extracted frames folder ::

        clusters inspect splitxyz0001

Preview a run using explicit cluster rules and periodic wrapping ::

        clusters preview splitxyz0001 \
            --use-pbc \
            --search-mode kdtree \
            --node Pb \
            --linker I \
            --shell O \
            --pair-cutoff Pb:I:3.36 \
            --pair-cutoff Pb:O:3.36

Export clusters without opening the UI ::

        clusters export splitxyz0001 \
            --use-pbc \
            --search-mode kdtree \
            --node Pb \
            --linker I \
            --shell O \
            --pair-cutoff Pb:I:3.36 \
            --pair-cutoff Pb:O:3.36

If the output folder already contains matching
``cluster_extraction_metadata.json`` data, the command will resume an
incomplete extraction or recognize that the extraction was already
completed.

The top-level ``saxshell`` command forwards to the same workflow ::

        saxshell cluster inspect splitxyz0001

Cluster Python and Notebook Use
-------------------------------

For notebook use, the ``ClusterWorkflow`` class exposes the same
inspection, preview, estimated-box handling, and export functionality
used by the UI and terminal interfaces.

::

        from saxshell.cluster import (
            ClusterWorkflow,
            example_atom_type_definitions,
            example_pair_cutoff_definitions,
        )

        workflow = ClusterWorkflow(
            "splitxyz0001",
            atom_type_definitions=example_atom_type_definitions(),
            pair_cutoff_definitions=example_pair_cutoff_definitions(),
            use_pbc=True,
            search_mode="kdtree",
        )
        summary = workflow.inspect()
        selection = workflow.preview_selection()
        result = workflow.export_clusters()

The resulting ``ClusterExportResult`` includes the metadata-file path
and resume status, so notebook code can inspect whether the export was
new, resumed, or already complete.

``bondanalysis``
~~~~~~~~~~~~~~~~

The ``bondanalysis`` application is designed for bond-pair and
angle-triplet distribution analysis on the stoichiometry-level cluster
folders produced by ``clusters``. A typical workflow is:

1. Use ``mdtrajectory`` to export frames.
2. Use ``clusters`` to sort those frames into stoichiometry folders.
3. Open ``bondanalysis`` on the cluster-output folder.
4. Choose the bond pairs and angle triplets to measure, along with
   their cutoff distances.
5. Run the analysis to write CSVs, histograms, comparison plots, and a
   JSON manifest into a sibling ``bondanalysis_*`` folder.

The same ``bondanalysis`` workflow can be used in three ways:

1. As a Qt desktop application for interactive use.
2. As a terminal command for scripted or batch workflows.
3. As a Python workflow in notebooks and other Python scripts.

To launch the Qt application, use one of the following commands ::

        bondanalysis
        bondanalysis ui
        saxshell bondanalysis
        python -m saxshell.bondanalysis

Bondanalysis UI Use
-------------------

The Qt interface focuses only on bond-pair and angle-distribution
analysis. The legacy displacement-analysis tooling is intentionally not
part of the new window and should be treated as deprecated until it is
updated.

Launch the UI ::

        bondanalysis

Or open the UI with a clusters folder preloaded ::

        bondanalysis ui clusters_splitxyz0001

Inside the window:

1. Select the top-level clusters directory that contains the
   stoichiometry folders.
2. Confirm the suggested sibling output directory, or choose a custom
   one.
3. Leave ``Analyze only checked cluster types`` off to process all
   stoichiometry folders, or turn it on and tick only the folders you
   want.
4. Use the ``Presets`` dropdown to load the built-in ``DMSO`` or
   ``DMF`` definitions, or save your current setup as a custom preset
   for a later session.
5. Add one or more bond-pair rows using ``Atom 1``, ``Atom 2``, and a
   distance cutoff in angstrom.
6. Add one or more angle-triplet rows using the vertex atom, the two
   arm atoms, and the two vertex-arm cutoffs in angstrom.
7. Click ``Analyze Bond Pairs and Angle Distributions``.

Each run writes:

1. Per-cluster-type CSVs, NPYs, and histogram PNGs under
   ``cluster_types/<cluster_type>/``.
2. Aggregate CSVs, NPYs, and histogram PNGs across all selected cluster
   types
   under ``all_clusters/``.
3. Overlay comparison CSVs, NPYs, and PNG plots under ``comparisons/``.
4. A ``bondanalysis_manifest.json`` file describing the run inputs and
   outputs.

Bondanalysis Terminal Use
-------------------------

Inspect a clusters directory before running analysis ::

        bondanalysis inspect clusters_splitxyz0001

Run bond-pair and angle analysis headlessly on every cluster type ::

        bondanalysis run clusters_splitxyz0001 \
            --bond-pair Pb:I:3.50 \
            --bond-pair Pb:O:3.20 \
            --angle-triplet Pb:I:I:3.50:3.50

Restrict the run to selected stoichiometry folders and choose an
explicit output directory ::

        bondanalysis run clusters_splitxyz0001 \
            --output-dir bondanalysis_clusters_splitxyz0001 \
            --cluster-type PbI2 \
            --cluster-type Pb2I3 \
            --bond-pair Pb:I:3.50 \
            --angle-triplet Pb:I:I:3.50:3.50

The top-level ``saxshell`` command forwards to the same workflow ::

        saxshell bondanalysis inspect clusters_splitxyz0001

Bondanalysis Python and Notebook Use
------------------------------------

For notebook use, the ``BondAnalysisWorkflow`` class exposes the same
inspection and execution steps used by the UI and terminal interfaces.

::

        from saxshell.bondanalysis import (
            AngleTripletDefinition,
            BondAnalysisWorkflow,
            BondPairDefinition,
        )

        workflow = BondAnalysisWorkflow(
            "clusters_splitxyz0001",
            bond_pairs=[
                BondPairDefinition("Pb", "I", 3.50),
                BondPairDefinition("Pb", "O", 3.20),
            ],
            angle_triplets=[
                AngleTripletDefinition("Pb", "I", "I", 3.50, 3.50),
            ],
        )
        summary = workflow.inspect()
        result = workflow.run()

The resulting ``BondAnalysisBatchResult`` records the selected cluster
types, output directory, processed file count, per-cluster summaries,
and manifest-file path.

Support and Contribute
----------------------

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/kewh5868/saxshell/issues>`_ and/or `submit a fix as a PR <https://github.com/kewh5868/saxshell/pulls>`_.

Feel free to fork the project and contribute. To install saxshell
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/kewh5868/saxshell/blob/main/CODE-OF-CONDUCT.rst>`_.

Contact
-------

For more information on saxshell please visit the project `web-page <https://kewh5868.github.io/>`_ or email the maintainers ``Keith White(keith.white@colorado.edu)``.

Acknowledgements
----------------

``saxshell`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.
