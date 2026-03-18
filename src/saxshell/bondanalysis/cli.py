from __future__ import annotations

import argparse
from pathlib import Path

from saxshell.version import __version__

from .bondanalyzer import AngleTripletDefinition, BondPairDefinition
from .workflow import BondAnalysisWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bondanalysis",
        description=(
            "Analyze bond-pair and angle-triplet distributions on "
            "stoichiometry-level cluster folders, or launch the Qt UI. "
            "Running without a subcommand launches the UI."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the bondanalysis version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Qt UI.")
    ui_parser.add_argument(
        "clusters_dir",
        nargs="?",
        type=Path,
        help="Optional clusters directory to prefill in the UI.",
    )
    ui_parser.set_defaults(handler=_handle_ui)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect the available cluster types in a clusters directory.",
    )
    inspect_parser.add_argument("clusters_dir", type=Path)
    inspect_parser.set_defaults(handler=_handle_inspect)

    run_parser = subparsers.add_parser(
        "run",
        help="Run bond-pair and angle-distribution analysis headlessly.",
    )
    run_parser.add_argument("clusters_dir", type=Path)
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory. Defaults to a suggested sibling folder.",
    )
    run_parser.add_argument(
        "--cluster-type",
        action="append",
        default=[],
        help=(
            "Restrict the run to this cluster type. Repeat as needed. "
            "Defaults to all discovered cluster types."
        ),
    )
    run_parser.add_argument(
        "--bond-pair",
        action="append",
        default=[],
        help="Bond pair definition as ATOM1:ATOM2:CUTOFF.",
    )
    run_parser.add_argument(
        "--angle-triplet",
        action="append",
        default=[],
        help=(
            "Angle triplet definition as " "VERTEX:ARM1:ARM2:CUTOFF1:CUTOFF2."
        ),
    )
    run_parser.set_defaults(handler=_handle_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"bondanalysis {__version__}")
        return 0

    if args.command is None:
        return _handle_ui(args)

    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _handle_ui(args: argparse.Namespace) -> int:
    from .ui.main_window import launch_bondanalysis_ui

    return launch_bondanalysis_ui(getattr(args, "clusters_dir", None))


def _handle_inspect(args: argparse.Namespace) -> int:
    workflow = BondAnalysisWorkflow(args.clusters_dir)
    summary = workflow.inspect()
    lines = [
        f"Clusters directory: {summary['clusters_dir']}",
        f"Cluster types detected: {summary['cluster_type_count']}",
        f"Total structure files: {summary['total_structure_files']}",
        f"Suggested output directory: {summary['suggested_output_dir']}",
    ]
    cluster_types = summary["cluster_types"]
    if cluster_types:
        lines.append("Cluster types: " + ", ".join(cluster_types))
    else:
        lines.append("Cluster types: none detected")
    print("\n".join(lines))
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    bond_pairs = _parse_bond_pairs(args.bond_pair)
    angle_triplets = _parse_angle_triplets(args.angle_triplet)
    workflow = BondAnalysisWorkflow(
        args.clusters_dir,
        bond_pairs=bond_pairs,
        angle_triplets=angle_triplets,
        output_dir=getattr(args, "output_dir", None),
        selected_cluster_types=(
            args.cluster_type if args.cluster_type else None
        ),
    )
    result = workflow.run()
    lines = [
        f"Clusters directory: {result.clusters_dir}",
        f"Output directory: {result.output_dir}",
        "Selected cluster types: " + ", ".join(result.selected_cluster_types),
        f"Structure files processed: {result.total_structure_files}",
        f"Manifest file: {result.manifest_path}",
    ]
    for cluster_result in result.cluster_results:
        bond_total = sum(cluster_result.bond_value_counts.values())
        angle_total = sum(cluster_result.angle_value_counts.values())
        lines.append(
            f"{cluster_result.cluster_type}: "
            f"{cluster_result.structure_count} file(s), "
            f"{bond_total} bond values, "
            f"{angle_total} angle values"
        )
    print("\n".join(lines))
    return 0


def _parse_bond_pairs(values: list[str]) -> list[BondPairDefinition]:
    definitions: list[BondPairDefinition] = []
    for raw in values:
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "Bond-pair arguments must look like ATOM1:ATOM2:CUTOFF."
            )
        definitions.append(
            BondPairDefinition(
                parts[0],
                parts[1],
                float(parts[2]),
            )
        )
    return definitions


def _parse_angle_triplets(
    values: list[str],
) -> list[AngleTripletDefinition]:
    definitions: list[AngleTripletDefinition] = []
    for raw in values:
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) != 5:
            raise ValueError(
                "Angle-triplet arguments must look like "
                "VERTEX:ARM1:ARM2:CUTOFF1:CUTOFF2."
            )
        definitions.append(
            AngleTripletDefinition(
                parts[0],
                parts[1],
                parts[2],
                float(parts[3]),
                float(parts[4]),
            )
        )
    return definitions


__all__ = ["build_parser", "main"]
