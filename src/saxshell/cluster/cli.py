from __future__ import annotations

import argparse
from pathlib import Path

from saxshell.version import __version__

from .clusternetwork import DEFAULT_SAVE_STATE_FREQUENCY, SEARCH_MODE_CHOICES
from .workflow import (
    ClusterExportResult,
    ClusterSelectionResult,
    ClusterWorkflow,
    example_atom_type_definitions,
    example_pair_cutoff_definitions,
    format_box_dimensions,
)


def _box_dimensions_line(summary: dict[str, object]) -> str:
    source_kind = summary.get("box_dimensions_source_kind")
    label = (
        "Source box dimensions"
        if source_kind == "source_filename"
        else "Estimated box dimensions"
    )
    box_dimensions = summary.get("box_dimensions")
    if box_dimensions is None:
        box_dimensions = summary.get("estimated_box_dimensions")
    return f"{label}: {format_box_dimensions(box_dimensions)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clusters",
        description=(
            "Inspect extracted cluster frame folders, preview box/PBC "
            "settings, export cluster files, or launch the Qt UI. Running "
            "without a subcommand launches the UI."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the clusters version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Qt UI.")
    ui_parser.add_argument(
        "frames_dir",
        nargs="?",
        type=Path,
        help="Optional extracted PDB or XYZ frames folder to prefill.",
    )
    ui_parser.set_defaults(handler=_handle_ui)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect an extracted PDB or XYZ frames folder.",
    )
    inspect_parser.add_argument("frames_dir", type=Path)
    inspect_parser.set_defaults(handler=_handle_inspect)

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview the current cluster-analysis selection.",
    )
    _add_common_cluster_arguments(preview_parser)
    preview_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory to preview. Defaults to a suggested "
        "folder beside the extracted frames directory.",
    )
    preview_parser.set_defaults(handler=_handle_preview)

    export_parser = subparsers.add_parser(
        "export",
        help="Run cluster analysis and export cluster files.",
    )
    _add_common_cluster_arguments(export_parser)
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory. Defaults to a suggested folder beside "
        "the extracted frames directory.",
    )
    export_parser.set_defaults(handler=_handle_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"clusters {__version__}")
        return 0

    if args.command is None:
        return _handle_ui(args)

    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _add_common_cluster_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Path to a folder of extracted single-frame .pdb or .xyz files.",
    )
    parser.add_argument(
        "--node",
        dest="node_rules",
        action="append",
        default=[],
        help="Node atom rule as ELEMENT or ELEMENT:RESIDUE.",
    )
    parser.add_argument(
        "--linker",
        dest="linker_rules",
        action="append",
        default=[],
        help="Linker atom rule as ELEMENT or ELEMENT:RESIDUE.",
    )
    parser.add_argument(
        "--shell",
        dest="shell_rules",
        action="append",
        default=[],
        help="Shell atom rule as ELEMENT or ELEMENT:RESIDUE.",
    )
    parser.add_argument(
        "--pair-cutoff",
        action="append",
        default=[],
        help=(
            "Pair cutoff as ELEMENT1:ELEMENT2:DISTANCE or "
            "ELEMENT1:ELEMENT2:SHELL_LEVEL:DISTANCE."
        ),
    )
    parser.add_argument(
        "--default-cutoff",
        type=float,
        help="Fallback cutoff used when a specific pair cutoff is missing.",
    )
    parser.add_argument(
        "--box",
        nargs=3,
        type=float,
        metavar=("LX", "LY", "LZ"),
        help="Manual box dimensions in angstrom.",
    )
    parser.add_argument(
        "--use-pbc",
        action="store_true",
        help="Enable periodic boundary conditions for clustering.",
    )
    parser.add_argument(
        "--search-mode",
        choices=SEARCH_MODE_CHOICES,
        default="kdtree",
        help=(
            "Neighbor-search mode for clustering. 'kdtree' and "
            "'vectorized' are faster than the legacy 'bruteforce' "
            "all-pairs loop."
        ),
    )
    parser.add_argument(
        "--save-state-frequency",
        type=int,
        default=DEFAULT_SAVE_STATE_FREQUENCY,
        help=(
            "Write resumable metadata after this many processed frames. "
            f"Defaults to {DEFAULT_SAVE_STATE_FREQUENCY}."
        ),
    )
    parser.add_argument(
        "--shell-level",
        dest="shell_levels",
        action="append",
        type=int,
        default=[],
        help="Grow this shell level around node atoms. Repeat as needed.",
    )
    parser.add_argument(
        "--include-shell-level",
        dest="include_shell_levels",
        action="append",
        type=int,
        default=[],
        help="Include this shell level in exported files. Repeat as needed.",
    )
    parser.add_argument(
        "--shared-shells",
        action="store_true",
        help="Allow shell atoms to be shared between clusters.",
    )
    parser.add_argument(
        "--legacy-solvation-shells",
        action="store_true",
        help=(
            "Disable Smart Solvation Shell mode and use the legacy "
            "per-frame solvent shell extraction path."
        ),
    )
    parser.add_argument(
        "--include-shell-atoms-in-stoichiometry",
        action="store_true",
        help="Include shell atoms when assigning stoichiometry folders.",
    )


def _handle_ui(args: argparse.Namespace) -> int:
    from .ui.main_window import launch_cluster_ui

    return launch_cluster_ui(getattr(args, "frames_dir", None))


def _build_workflow(args: argparse.Namespace) -> ClusterWorkflow:
    return ClusterWorkflow(
        frames_dir=args.frames_dir,
        atom_type_definitions=_parse_atom_type_definitions(args),
        pair_cutoff_definitions=_parse_pair_cutoffs(args),
        box_dimensions=tuple(args.box) if args.box is not None else None,
        use_pbc=bool(getattr(args, "use_pbc", False)),
        default_cutoff=getattr(args, "default_cutoff", None),
        shell_levels=_parse_shell_levels(args.shell_levels),
        include_shell_levels=_parse_include_shell_levels(
            args.include_shell_levels
        ),
        search_mode=str(getattr(args, "search_mode", "kdtree")),
        save_state_frequency=int(
            getattr(
                args,
                "save_state_frequency",
                DEFAULT_SAVE_STATE_FREQUENCY,
            )
        ),
        shared_shells=bool(getattr(args, "shared_shells", False)),
        smart_solvation_shells=not bool(
            getattr(args, "legacy_solvation_shells", False)
        ),
        include_shell_atoms_in_stoichiometry=bool(
            getattr(args, "include_shell_atoms_in_stoichiometry", False)
        ),
    )


def _parse_atom_type_definitions(
    args: argparse.Namespace,
) -> dict[str, list[tuple[str, str | None]]]:
    parsed = {
        "node": _parse_atom_rule_list(args.node_rules),
        "linker": _parse_atom_rule_list(args.linker_rules),
        "shell": _parse_atom_rule_list(args.shell_rules),
    }
    if any(parsed.values()):
        return {key: value for key, value in parsed.items() if value}
    return example_atom_type_definitions()


def _parse_atom_rule_list(
    values: list[str],
) -> list[tuple[str, str | None]]:
    rules: list[tuple[str, str | None]] = []
    for value in values:
        parts = [part.strip() for part in value.split(":", maxsplit=1)]
        if not parts[0]:
            raise ValueError(f"Invalid atom rule: {value!r}")
        residue = parts[1] if len(parts) == 2 and parts[1] else None
        rules.append((parts[0].title(), residue))
    return rules


def _parse_pair_cutoffs(
    args: argparse.Namespace,
) -> dict[tuple[str, str], dict[int, float]]:
    if not args.pair_cutoff:
        return example_pair_cutoff_definitions()

    parsed: dict[tuple[str, str], dict[int, float]] = {}
    for value in args.pair_cutoff:
        parts = [part.strip() for part in value.split(":")]
        if len(parts) == 3:
            atom1, atom2, cutoff_text = parts
            shell_level = 0
        elif len(parts) == 4:
            atom1, atom2, shell_text, cutoff_text = parts
            shell_level = int(shell_text)
        else:
            raise ValueError(
                "Pair cutoffs must be ELEMENT1:ELEMENT2:DISTANCE or "
                "ELEMENT1:ELEMENT2:SHELL_LEVEL:DISTANCE."
            )
        cutoff = float(cutoff_text)
        parsed.setdefault((atom1.title(), atom2.title()), {})[
            shell_level
        ] = cutoff
    return parsed


def _parse_shell_levels(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values if value > 0}))


def _parse_include_shell_levels(values: list[int]) -> tuple[int, ...]:
    include_levels = {0}
    include_levels.update(int(value) for value in values if value >= 0)
    return tuple(sorted(include_levels))


def _handle_inspect(args: argparse.Namespace) -> int:
    workflow = ClusterWorkflow(
        frames_dir=args.frames_dir,
        atom_type_definitions=example_atom_type_definitions(),
        pair_cutoff_definitions=example_pair_cutoff_definitions(),
    )
    summary = workflow.inspect()
    lines = [
        f"Frames folder: {args.frames_dir}",
        f"Mode: {summary['mode_label']}",
        f"Frames: {summary['n_frames']}",
        f"Output format: {summary['output_file_extension']}",
        _box_dimensions_line(summary),
    ]
    if summary.get("box_dimensions_source") is not None:
        lines.append(f"Box source: {summary['box_dimensions_source']}")
    print("\n".join(lines))
    return 0


def _handle_preview(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    selection = workflow.preview_selection(output_dir=args.output_dir)
    print(_format_selection_result(selection))
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    result = workflow.export_clusters(output_dir=args.output_dir)
    print(_format_export_result(result))
    return 0


def _format_selection_result(selection: ClusterSelectionResult) -> str:
    stoichiometry_bins_text = (
        "solute + shell atoms"
        if selection.include_shell_atoms_in_stoichiometry
        else "solute only"
    )
    lines = [
        f"Frames folder: {selection.summary['input_dir']}",
        f"Mode: {selection.mode_label}",
        f"PBC: {'on' if selection.use_pbc else 'off'}",
        "Smart solvation shells: "
        f"{'on' if selection.smart_solvation_shells else 'off'}",
        f"Search mode: {selection.search_mode_label}",
        "Save-state frequency: "
        f"every {selection.save_state_frequency} frames",
        f"Stoichiometry bins: {stoichiometry_bins_text}",
        f"Frames selected: {selection.total_frames}",
        "Frame file range: "
        f"{selection.summary.get('first_frame')} to "
        f"{selection.summary.get('last_frame')}",
        f"Output format: {selection.output_file_extension}",
        f"Output directory: {selection.output_dir}",
        _box_dimensions_line(selection.summary),
    ]
    if selection.box_dimensions_source is not None:
        lines.append(f"Box source: {selection.box_dimensions_source}")
    if selection.resolved_box_dimensions is None:
        lines.append("Resolved box dimensions: not used")
    elif selection.using_auto_box:
        lines.append(
            "Resolved box dimensions: "
            f"{format_box_dimensions(selection.resolved_box_dimensions)} "
            "(auto)"
        )
    else:
        lines.append(
            "Resolved box dimensions: "
            f"{format_box_dimensions(selection.resolved_box_dimensions)}"
        )
    supports_full_molecules = bool(
        selection.summary.get("supports_full_molecule_shells", False)
    )
    lines.append(f"Supports full molecule shells: {supports_full_molecules}")
    return "\n".join(lines)


def _format_export_result(result: ClusterExportResult) -> str:
    lines = [
        _format_selection_result(result.selection),
        f"Frames analyzed: {result.analyzed_frames}",
        f"Clusters found: {result.total_clusters}",
        f"Files written: {len(result.written_files)}",
    ]
    if result.already_complete:
        lines.append(
            "Resume status: extraction already complete; reused existing "
            "results"
        )
    elif result.resumed:
        lines.append(
            "Resume status: resumed existing extraction "
            f"({result.previously_completed_frames} frame(s) were already "
            "complete)"
        )
    else:
        lines.append("Resume status: new extraction")
    lines.append("Newly processed frames: " f"{result.newly_processed_frames}")
    if result.metadata_path is not None:
        lines.append(f"Metadata file: {result.metadata_path}")
    if result.written_files:
        lines.append(f"First written file: {result.written_files[0]}")
    return "\n".join(lines)
