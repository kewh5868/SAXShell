from __future__ import annotations

import argparse
import sys
from pathlib import Path

from saxshell.version import __version__

from .frame.manager import DEFAULT_FRAME_TIMESTEP_FS
from .workflow import (
    MDTrajectoryAssertionResult,
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
    _add_restart_duplicate_argument(inspect_parser)
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
    _add_restart_duplicate_argument(preview_parser)
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
    _add_restart_duplicate_argument(export_parser)
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory. Defaults to a new suggested folder "
        "next to the trajectory.",
    )
    export_parser.set_defaults(handler=_handle_export)

    validate_parser = subparsers.add_parser(
        "validate-export",
        help=(
            "Assert that exported XYZ frames map back to the source "
            "trajectory indices and coordinates."
        ),
    )
    validate_parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to the source trajectory file (.xyz).",
    )
    validate_parser.add_argument(
        "frame_dir",
        type=Path,
        help="Directory containing exported frame_<index>.xyz files.",
    )
    validate_parser.add_argument(
        "--coordinate-lines",
        type=int,
        default=3,
        help="Number of leading coordinate lines to compare. Default: 3.",
    )
    validate_parser.add_argument(
        "--coord-tol",
        type=float,
        default=1.0e-9,
        help="Absolute coordinate comparison tolerance. Default: 1e-9.",
    )
    validate_parser.add_argument(
        "--expect-contiguous",
        action="store_true",
        help="Fail if exported filename indices have gaps within their range.",
    )
    validate_parser.add_argument(
        "--strict-source-duplicates",
        action="store_true",
        help="Fail if the source trajectory contains duplicate i = indices.",
    )
    validate_parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help="Maximum number of issue examples to print. Default: 20.",
    )
    validate_parser.set_defaults(handler=_handle_validate_export)

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
    parser.add_argument(
        "--frame-timestep-fs",
        type=float,
        default=DEFAULT_FRAME_TIMESTEP_FS,
        help=(
            "Fallback frame timestep in femtoseconds when trajectory frames "
            f"do not include source times. Default: {DEFAULT_FRAME_TIMESTEP_FS:g}."
        ),
    )
    parser.add_argument(
        "--manual-frame-timestep",
        action="store_true",
        help=(
            "Use --frame-timestep-fs for all frame times instead of source "
            "trajectory time metadata."
        ),
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
        "--frame-interval",
        dest="stride",
        type=int,
        default=1,
        help="Frame interval: keep every Nth frame. Default: 1.",
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
        default=2,
        help=(
            "Consecutive sample window used for the steady-state test. "
            "Default: 2."
        ),
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


def _add_restart_duplicate_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-restart-duplicates",
        action="store_true",
        help=(
            "Include duplicate XYZ frames from overlapping simulation "
            "restarts. By default, earlier overlap frames are skipped and "
            "the later continuation frame is kept."
        ),
    )


def _handle_ui(_: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from saxshell.saxs.ui.branding import prepare_saxshell_application_identity

    from .ui.main_window import launch_mdtrajectory_app

    app = QApplication.instance()
    created_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
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
        include_restart_duplicates=getattr(
            args,
            "include_restart_duplicates",
            False,
        ),
        frame_timestep_fs=getattr(
            args,
            "frame_timestep_fs",
            DEFAULT_FRAME_TIMESTEP_FS,
        ),
        use_inferred_frame_times=getattr(
            args,
            "manual_frame_timestep",
            False,
        ),
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
            window=getattr(args, "window", 2),
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
    if "raw_frames" in summary:
        include_duplicates = bool(
            summary.get("include_restart_duplicates", False)
        )
        duplicate_action = "included" if include_duplicates else "skipped"
        lines.extend(
            [
                f"Raw frames: {summary['raw_frames']}",
                f"Duplicate source frames {duplicate_action}: "
                f"{summary.get('duplicate_source_frames', 0)}",
            ]
        )
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
    detected_timestep = summary.get("detected_frame_timestep_fs")
    if detected_timestep is not None:
        lines.append(
            f"Detected frame timestep: {float(detected_timestep):.6g} fs"
        )
    elif summary.get("frame_timestep_fs") is not None:
        lines.append(
            "Fallback frame timestep: "
            f"{float(summary['frame_timestep_fs']):.6g} fs"
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


def _handle_validate_export(args: argparse.Namespace) -> int:
    workflow = _build_workflow(args)
    result = workflow.validate_export(
        args.frame_dir,
        coordinate_lines=args.coordinate_lines,
        coordinate_tolerance=args.coord_tol,
        expect_contiguous=args.expect_contiguous,
        strict_source_duplicates=args.strict_source_duplicates,
        max_issues=args.max_issues,
    )
    print(_format_assertion_result(result))
    return 0 if result.passed else 1


def _format_selection_result(selection: MDTrajectorySelectionResult) -> str:
    preview = selection.preview
    lines = [
        f"Output directory: {selection.output_dir}",
        f"Frames selected: {preview.selected_frames}",
        f"Trajectory frames: {preview.total_frames}",
        f"Start: {preview.start}",
        f"Stop: {preview.stop}",
        f"Frame interval: {preview.stride}",
        f"Time-tagged frames: {preview.time_metadata_frames}",
        "Restart duplicate frames: "
        f"{'included' if selection.include_restart_duplicates else 'skipped'}",
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


def _format_assertion_result(result: MDTrajectoryAssertionResult) -> str:
    status = "passed" if result.passed else "failed"
    lines = [
        f"Export validation {status}.",
        f"Trajectory file: {result.trajectory_file}",
        f"Frame directory: {result.frame_dir}",
        f"Coordinate lines checked: {result.coordinate_lines}",
        f"Coordinate tolerance: {result.coordinate_tolerance:g}",
        f"Source raw frames: {result.source_raw_frames}",
        f"Source unique indices: {result.source_unique_indices}",
        f"Source frames without i index: {result.source_missing_indices}",
        f"Source duplicate i indices: {result.source_duplicate_indices}",
        "Source duplicate coordinate conflicts: "
        f"{result.source_duplicate_conflicts}",
        f"Exported XYZ files: {result.exported_files}",
        f"Validated XYZ files: {result.validated_files}",
    ]
    if result.filename_index_min is not None:
        lines.append(
            "Filename index range: "
            f"{result.filename_index_min} to {result.filename_index_max}"
        )
    if result.header_index_min is not None:
        lines.append(
            "Header index range: "
            f"{result.header_index_min} to {result.header_index_max}"
        )
    if result.filename_header_offsets:
        offsets = ", ".join(
            f"{offset}: {count}"
            for offset, count in result.filename_header_offsets.items()
        )
        lines.append(f"Filename-header offsets: {offsets}")
    else:
        lines.append("Filename-header offsets: none")

    if result.issue_counts:
        lines.append("Assertion failures:")
        lines.extend(
            f"- {kind}: {count}" for kind, count in result.issue_counts.items()
        )
    else:
        lines.append("Assertion failures: none")

    if result.strict_source_duplicates and result.source_duplicate_indices:
        lines.append(
            "Strict source duplicate check failed: "
            f"{result.source_duplicate_indices} duplicate source frame(s)."
        )

    if result.issues:
        lines.append("Issue examples:")
        for issue in result.issues:
            location = "" if issue.path is None else f" [{issue.path}]"
            lines.append(f"- {issue.kind}{location}: {issue.message}")

    return "\n".join(lines)
