from __future__ import annotations

import argparse
import sys
from pathlib import Path

from saxshell.version import __version__

from .workflow import (
    MDTrajectoryExportResult,
    MDTrajectorySelectionResult,
    MDTrajectoryWorkflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdtrajectory",
        description=(
            "Inspect MD trajectories, suggest a CP2K cutoff, preview frame "
            "selection, export frames, or launch the Qt UI. Running "
            "without a subcommand launches the UI."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the mdtrajectory version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Qt UI.")
    ui_parser.set_defaults(handler=_handle_ui)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect the trajectory and optionally the CP2K energy file.",
    )
    _add_common_input_arguments(inspect_parser)
    inspect_parser.set_defaults(handler=_handle_inspect)

    suggest_parser = subparsers.add_parser(
        "suggest-cutoff",
        help="Suggest a steady-state cutoff from a CP2K energy file.",
    )
    _add_common_input_arguments(suggest_parser)
    _add_cutoff_analysis_arguments(suggest_parser, required_target=True)
    suggest_parser.set_defaults(handler=_handle_suggest_cutoff)

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview the current frame selection and output directory.",
    )
    _add_common_input_arguments(preview_parser)
    _add_selection_arguments(preview_parser)
    _add_cutoff_resolution_arguments(preview_parser)
    preview_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory to preview. Defaults to the suggested "
        "output folder next to the trajectory.",
    )
    preview_parser.set_defaults(handler=_handle_preview)

    export_parser = subparsers.add_parser(
        "export",
        help="Export the selected frames without launching the UI.",
    )
    _add_common_input_arguments(export_parser)
    _add_selection_arguments(export_parser)
    _add_cutoff_resolution_arguments(export_parser)
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory. Defaults to a new suggested folder "
        "next to the trajectory.",
    )
    export_parser.set_defaults(handler=_handle_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"mdtrajectory {__version__}")
        return 0

    if args.command is None:
        return _handle_ui(args)

    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _add_common_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to the trajectory file (.xyz or .pdb).",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        help="Optional topology file for formats that require one.",
    )
    parser.add_argument(
        "--energy-file",
        type=Path,
        help="Optional CP2K .ener file for cutoff analysis.",
    )


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--start",
        type=int,
        help="Zero-based start frame index.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        help="Exclusive stop frame index.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame. Default: 1.",
    )


def _add_cutoff_analysis_arguments(
    parser: argparse.ArgumentParser,
    *,
    required_target: bool,
) -> None:
    parser.add_argument(
        "--temp-target-k",
        type=float,
        required=required_target,
        help="Target temperature in Kelvin for steady-state detection.",
    )
    parser.add_argument(
        "--temp-tol-k",
        type=float,
        default=1.0,
        help="Allowed temperature deviation in Kelvin. Default: 1.0.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Consecutive sample window used for the steady-state test.",
    )


def _add_cutoff_resolution_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--use-cutoff",
        action="store_true",
        help="Apply a cutoff to the preview or export selection.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cutoff-fs",
        type=float,
        help="Manual cutoff time in femtoseconds.",
    )
    group.add_argument(
        "--use-suggested-cutoff",
        action="store_true",
        help="Run the cutoff analyzer and apply the suggested cutoff.",
    )
    _add_cutoff_analysis_arguments(parser, required_target=False)


def _handle_ui(_: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import launch_mdtrajectory_app

    app = QApplication.instance()
    created_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    launch_mdtrajectory_app()
    if created_app:
        assert app is not None
        return int(app.exec())
    return 0


def _build_workflow(args: argparse.Namespace) -> MDTrajectoryWorkflow:
    return MDTrajectoryWorkflow(
        trajectory_file=args.trajectory,
        topology_file=getattr(args, "topology", None),
        energy_file=getattr(args, "energy_file", None),
    )


def _resolve_cli_cutoff(
    workflow: MDTrajectoryWorkflow,
    args: argparse.Namespace,
) -> tuple[bool, float | None]:
    use_cutoff = bool(
        getattr(args, "use_cutoff", False)
        or getattr(args, "cutoff_fs", None) is not None
        or getattr(args, "use_suggested_cutoff", False)
    )
    cutoff_fs = getattr(args, "cutoff_fs", None)

    if getattr(args, "use_suggested_cutoff", False):
        temp_target_k = getattr(args, "temp_target_k", None)
        if temp_target_k is None:
            raise ValueError(
                "--temp-target-k is required with --use-suggested-cutoff."
            )
        result = workflow.suggest_cutoff(
            temp_target_k=temp_target_k,
            temp_tol_k=getattr(args, "temp_tol_k", 1.0),
            window=getattr(args, "window", 3),
        )
        cutoff_fs = result.cutoff_time_fs
        if cutoff_fs is None:
            raise ValueError(
                "No steady-state cutoff could be suggested from the "
                "provided CP2K energy file."
            )
        workflow.set_selected_cutoff(cutoff_fs)
    elif cutoff_fs is not None:
        workflow.set_selected_cutoff(cutoff_fs)

    return use_cutoff, cutoff_fs


def _handle_inspect(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    summary = workflow.inspect()
    lines = [
        f"Trajectory file: {workflow.trajectory_file}",
        f"File type: {summary['file_type']}",
        f"Frames: {summary['n_frames']}",
    ]
    if workflow.topology_file is not None:
        lines.append(f"Topology file: {workflow.topology_file}")
    if workflow.energy_file is not None:
        energy_data = workflow.load_energy()
        lines.extend(
            [
                f"Energy file: {workflow.energy_file}",
                f"Energy samples: {energy_data.n_points}",
                "Energy time range: "
                f"{energy_data.time_min_fs:.3f} fs to "
                f"{energy_data.time_max_fs:.3f} fs",
            ]
        )
    print("\n".join(lines))
    return 0


def _handle_suggest_cutoff(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    workflow.inspect()
    result = workflow.suggest_cutoff(
        temp_target_k=args.temp_target_k,
        temp_tol_k=args.temp_tol_k,
        window=args.window,
    )
    if result.cutoff_time_fs is None:
        print("Suggested cutoff: None")
        return 0

    workflow.set_selected_cutoff(result.cutoff_time_fs)
    lines = [
        f"Suggested cutoff: {result.cutoff_time_fs:.3f} fs",
        f"Target temperature: {result.temp_target_k:.3f} K",
        f"Temperature tolerance: {result.temp_tol_k:.3f} K",
        f"Window: {result.window}",
    ]
    print("\n".join(lines))
    return 0


def _handle_preview(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    workflow.inspect()
    use_cutoff, cutoff_fs = _resolve_cli_cutoff(workflow, args)
    selection = workflow.preview_selection(
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        use_cutoff=use_cutoff,
        cutoff_fs=cutoff_fs,
        output_dir=args.output_dir,
    )
    print(_format_selection_result(selection))
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    workflow.inspect()
    use_cutoff, cutoff_fs = _resolve_cli_cutoff(workflow, args)
    result = workflow.export_frames(
        output_dir=args.output_dir,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        use_cutoff=use_cutoff,
        cutoff_fs=cutoff_fs,
    )
    print(_format_export_result(result))
    return 0


def _format_selection_result(selection: MDTrajectorySelectionResult) -> str:
    preview = selection.preview
    lines = [
        f"Output directory: {selection.output_dir}",
        f"Frames selected: {preview.selected_frames}",
        f"Trajectory frames: {preview.total_frames}",
        f"Start: {preview.start}",
        f"Stop: {preview.stop}",
        f"Stride: {preview.stride}",
        f"Time-tagged frames: {preview.time_metadata_frames}",
    ]
    if selection.applied_cutoff_fs is not None:
        lines.append(f"Applied cutoff: {selection.applied_cutoff_fs:.3f} fs")
    else:
        lines.append("Applied cutoff: None")
    if preview.first_frame_index is not None:
        lines.append(
            "Frame index range: "
            f"{preview.first_frame_index} to {preview.last_frame_index}"
        )
    if preview.first_time_fs is not None:
        lines.append(
            "Time range: "
            f"{preview.first_time_fs:.3f} fs to "
            f"{preview.last_time_fs:.3f} fs"
        )
    return "\n".join(lines)


def _format_export_result(result: MDTrajectoryExportResult) -> str:
    lines = [
        "Frame export complete.",
        _format_selection_result(result.selection),
        f"Written files: {len(result.written_files)}",
    ]
    if result.written_files:
        lines.append(f"First file: {result.written_files[0]}")
        lines.append(f"Last file: {result.written_files[-1]}")
    return "\n".join(lines)
