from __future__ import annotations

import argparse
import sys
from pathlib import Path

from saxshell.version import __version__

from .workflow import (
    XYZToPDBWorkflow,
    create_reference_molecule,
    default_reference_library_dir,
    list_reference_library,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xyz2pdb",
        description=(
            "Convert one XYZ file or a folder of XYZ files into PDB files "
            "using reference-molecule PDBs and residue-assignment rules. "
            "Running without a subcommand launches the Qt UI."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the xyz2pdb version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Qt UI.")
    ui_parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Optional XYZ file or folder to prefill.",
    )
    ui_parser.add_argument(
        "--config",
        type=Path,
        help="Optional residue-assignment JSON file to prefill.",
    )
    ui_parser.add_argument(
        "--library-dir",
        type=Path,
        help="Optional reference-library directory to prefill.",
    )
    ui_parser.set_defaults(handler=_handle_ui)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect the XYZ input, reference library, and optional config.",
    )
    _add_common_conversion_arguments(
        inspect_parser,
        require_config=False,
        include_output_dir=False,
    )
    inspect_parser.set_defaults(handler=_handle_inspect)

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview the first-frame residue assignments and output folder.",
    )
    _add_common_conversion_arguments(
        preview_parser,
        require_config=True,
        include_output_dir=True,
    )
    preview_parser.set_defaults(handler=_handle_preview)

    export_parser = subparsers.add_parser(
        "export",
        help="Convert the selected XYZ input into PDB files.",
    )
    _add_common_conversion_arguments(
        export_parser,
        require_config=True,
        include_output_dir=True,
    )
    export_parser.set_defaults(handler=_handle_export)

    references_parser = subparsers.add_parser(
        "references",
        help="List bundled reference molecules or add a new one.",
    )
    references_subparsers = references_parser.add_subparsers(
        dest="reference_command"
    )

    list_parser = references_subparsers.add_parser(
        "list",
        help="List the available reference PDBs in one library folder.",
    )
    list_parser.add_argument(
        "--library-dir",
        type=Path,
        help="Reference-library directory. Defaults to the bundled folder.",
    )
    list_parser.set_defaults(handler=_handle_reference_list)

    add_parser = references_subparsers.add_parser(
        "add",
        help="Create a new reference PDB from a source PDB or XYZ file.",
    )
    add_parser.add_argument("source_file", type=Path)
    add_parser.add_argument(
        "--name",
        required=True,
        help="Filename stem to use for the new reference PDB.",
    )
    add_parser.add_argument(
        "--residue-name",
        help="Optional residue name written into the saved reference PDB.",
    )
    add_parser.add_argument(
        "--library-dir",
        type=Path,
        help="Reference-library directory. Defaults to the bundled folder.",
    )
    add_parser.set_defaults(handler=_handle_reference_add)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"xyz2pdb {__version__}")
        return 0

    if (
        args.command == "references"
        and getattr(args, "reference_command", None) is None
    ):
        parser.exit(
            2, "Error: choose 'references list' or 'references add'.\n"
        )

    if args.command is None:
        return _handle_ui(args)

    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _add_common_conversion_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_config: bool,
    include_output_dir: bool,
) -> None:
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to one XYZ file or a folder of XYZ files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=require_config,
        help="Residue-assignment JSON file describing molecules, anchors, and free atoms.",
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        help="Reference-library directory. Defaults to the bundled folder.",
    )
    if include_output_dir:
        parser.add_argument(
            "--output-dir",
            type=Path,
            help="Output directory for converted PDB files.",
        )


def _handle_ui(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from saxshell.saxs.ui.branding import prepare_saxshell_application_identity

    from .ui.main_window import launch_xyz2pdb_ui

    app = QApplication.instance()
    created_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    launch_xyz2pdb_ui(
        input_path=getattr(args, "input_path", None),
        config_file=getattr(args, "config", None),
        reference_library_dir=getattr(args, "library_dir", None),
    )
    if created_app:
        assert app is not None
        return int(app.exec())
    return 0


def _build_workflow(args: argparse.Namespace) -> XYZToPDBWorkflow:
    return XYZToPDBWorkflow(
        input_path=args.input_path,
        config_file=getattr(args, "config", None),
        reference_library_dir=getattr(args, "library_dir", None),
        output_dir=getattr(args, "output_dir", None),
    )


def _handle_inspect(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    inspection = workflow.inspect()
    lines = [
        f"Input path: {inspection.input_path}",
        (
            "Input mode: Single XYZ file"
            if inspection.input_mode == "single_xyz"
            else "Input mode: XYZ folder"
        ),
        f"XYZ files: {inspection.total_files}",
        f"Reference library: {inspection.reference_library_dir}",
        "Available references: "
        + (
            ", ".join(entry.name for entry in inspection.available_references)
            if inspection.available_references
            else "none"
        ),
    ]
    if inspection.first_file is not None:
        lines.append(f"First XYZ file: {inspection.first_file}")
    if inspection.config_file is not None:
        lines.extend(
            [
                f"Config file: {inspection.config_file}",
                "Configured molecules: "
                + (
                    ", ".join(inspection.configured_molecules)
                    if inspection.configured_molecules
                    else "none"
                ),
                "Configured references: "
                + (
                    ", ".join(inspection.configured_reference_names)
                    if inspection.configured_reference_names
                    else "none"
                ),
            ]
        )
        if inspection.free_atom_elements:
            lines.append(
                "Free-atom assignments: "
                + ", ".join(inspection.free_atom_elements)
            )
    print("\n".join(lines))
    return 0


def _handle_preview(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    preview = workflow.preview_conversion(output_dir=args.output_dir)
    lines = [
        f"Input path: {preview.inspection.input_path}",
        f"Output directory: {preview.output_dir}",
        f"XYZ files to convert: {preview.inspection.total_files}",
        f"Configured references: {', '.join(preview.inspection.configured_reference_names)}",
        f"Total residues in first frame: {len(preview.residues)}",
        f"Total atoms in first frame: {preview.total_atoms}",
    ]
    if preview.first_output_file is not None:
        lines.append(f"First output file: {preview.first_output_file}")
    lines.append(
        "Detected molecules: "
        + ", ".join(
            f"{name} x{count}"
            for name, count in sorted(preview.molecule_counts.items())
        )
    )
    lines.append(
        "Residues to write: "
        + ", ".join(
            f"{name} x{count}"
            for name, count in sorted(preview.residue_counts.items())
        )
    )
    print("\n".join(lines))
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    result = workflow.export_pdbs(output_dir=args.output_dir)
    print("XYZ to PDB conversion complete.")
    print(f"Output directory: {result.output_dir}")
    print(f"Files written: {len(result.written_files)}")
    if result.written_files:
        print(f"First file: {result.written_files[0]}")
        print(f"Last file: {result.written_files[-1]}")
    print(
        "First-frame molecules: "
        + ", ".join(
            f"{name} x{count}"
            for name, count in sorted(result.preview.molecule_counts.items())
        )
    )
    return 0


def _handle_reference_list(args: argparse.Namespace) -> int:
    library_dir = (
        default_reference_library_dir()
        if getattr(args, "library_dir", None) is None
        else Path(args.library_dir)
    )
    entries = list_reference_library(library_dir)
    print(f"Reference library: {library_dir}")
    if not entries:
        print("No reference molecules found.")
        return 0
    for entry in entries:
        print(
            f"{entry.name}: residue {entry.residue_name}, "
            f"{entry.atom_count} atoms, {entry.path.name}"
        )
    return 0


def _handle_reference_add(args: argparse.Namespace) -> int:
    result = create_reference_molecule(
        args.source_file,
        reference_name=args.name,
        residue_name=getattr(args, "residue_name", None),
        library_dir=getattr(args, "library_dir", None),
    )
    print(f"Reference created: {result.name}")
    print(f"Output path: {result.path}")
    print(f"Residue name: {result.residue_name}")
    print(f"Atom count: {result.atom_count}")
    return 0
