from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from saxshell.version import __version__

from .run_config import (
    default_clusterdynamics_run_file_path,
    load_clusterdynamics_run_config,
    run_clusterdynamics_run_config,
)

_COMMANDS = {"setup-ui", "ui", "run", "batch-run"}
_TOP_LEVEL_OPTIONS = {"-h", "--help", "--version"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clusterdynamics",
        description=(
            "Analyze time-binned cluster distributions from extracted PDB "
            "or XYZ frame folders. Running without a subcommand launches "
            "the Qt UI."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the clusterdynamics version number and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    setup_ui_parser = subparsers.add_parser(
        "setup-ui",
        help="Launch the beta project-backed run-file setup interface.",
    )
    setup_ui_parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        help="Optional SAXSShell project folder.",
    )
    setup_ui_parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional extracted frames folder to prefill.",
    )
    setup_ui_parser.add_argument(
        "--energy-file",
        type=Path,
        default=None,
        help="Optional CP2K .ener file to prefill.",
    )
    setup_ui_parser.set_defaults(handler=_handle_setup_ui)

    ui_parser = subparsers.add_parser("ui", help="Launch the Qt UI.")
    ui_parser.add_argument(
        "frames_dir",
        nargs="?",
        type=Path,
        help="Optional extracted frames directory to prefill in the UI.",
    )
    ui_parser.add_argument(
        "--energy-file",
        type=Path,
        default=None,
        help="Optional CP2K .ener file to prefill in the UI.",
    )
    ui_parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Optional SAXSShell project directory to prefill in the UI.",
    )
    ui_parser.set_defaults(handler=_handle_ui)

    run_parser = subparsers.add_parser(
        "run",
        help="Run cluster dynamics from a project-backed run file.",
    )
    run_parser.add_argument(
        "project_dir",
        type=Path,
        help="SAXSShell project folder containing the run file.",
    )
    run_parser.add_argument(
        "--run-file",
        type=Path,
        default=None,
        help=(
            "Run file path. Defaults to cluster_dynamics_cli_run.json "
            "in the project folder."
        ),
    )
    run_parser.set_defaults(handler=_handle_run)

    batch_parser = subparsers.add_parser(
        "batch-run",
        help="Run the default cluster dynamics run file for multiple projects.",
    )
    batch_parser.add_argument(
        "project_dirs",
        nargs="+",
        type=Path,
        help="SAXSShell project folders to process.",
    )
    batch_parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running later projects if one project fails.",
    )
    batch_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of project runs to execute concurrently.",
    )
    batch_parser.set_defaults(handler=_handle_batch_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or raw_args[0] not in _COMMANDS | _TOP_LEVEL_OPTIONS:
        return _handle_legacy_ui(raw_args)

    parser = build_parser()
    args = parser.parse_args(raw_args)
    if args.version:
        print(f"clusterdynamics {__version__}")
        return 0
    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _handle_legacy_ui(raw_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="clusterdynamics",
        description="Launch the SAXSShell clusterdynamics UI.",
    )
    parser.add_argument(
        "frames_dir",
        nargs="?",
        help="Optional extracted frames directory to prefill in the UI.",
    )
    parser.add_argument(
        "--energy-file",
        help="Optional CP2K .ener file to prefill in the UI.",
    )
    parser.add_argument(
        "--project-dir",
        help="Optional SAXSShell project directory to prefill in the UI.",
    )
    args = parser.parse_args(raw_args)
    return _launch_ui(
        getattr(args, "frames_dir", None),
        energy_file=getattr(args, "energy_file", None),
        project_dir=getattr(args, "project_dir", None),
    )


def _handle_setup_ui(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.run_file_window import launch_clusterdynamics_run_file_ui

    owns_app = QApplication.instance() is None
    launch_clusterdynamics_run_file_ui(
        initial_project_dir=getattr(args, "project_dir", None),
        initial_frames_dir=getattr(args, "frames_dir", None),
        initial_energy_file=getattr(args, "energy_file", None),
    )
    app = QApplication.instance()
    if owns_app and app is not None:
        return app.exec()
    return 0


def _handle_ui(args: argparse.Namespace) -> int:
    return _launch_ui(
        getattr(args, "frames_dir", None),
        energy_file=getattr(args, "energy_file", None),
        project_dir=getattr(args, "project_dir", None),
    )


def _launch_ui(
    frames_dir: str | Path | None = None,
    *,
    energy_file: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> int:
    from .ui.main_window import launch_clusterdynamics_ui

    return launch_clusterdynamics_ui(
        frames_dir,
        energy_file=energy_file,
        project_dir=project_dir,
    )


def _handle_run(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).expanduser().resolve()
    run_file = _resolve_run_file(project_dir, args.run_file)
    config = load_clusterdynamics_run_config(run_file)
    summary = run_clusterdynamics_run_config(
        project_dir,
        config,
        run_file_path=run_file,
        log_callback=print,
        progress_callback=_print_progress,
    )
    _print_summary(summary)
    return 0


def _handle_batch_run(args: argparse.Namespace) -> int:
    workers = max(int(getattr(args, "workers", 1)), 1)
    project_dirs = [
        Path(project_dir_value).expanduser().resolve()
        for project_dir_value in args.project_dirs
    ]
    if workers > 1:
        return _handle_parallel_batch_run(
            project_dirs,
            workers=workers,
            keep_going=bool(args.keep_going),
        )

    failures: list[tuple[Path, str]] = []
    for project_dir in project_dirs:
        try:
            run_file = default_clusterdynamics_run_file_path(project_dir)
            config = load_clusterdynamics_run_config(run_file)
            summary = run_clusterdynamics_run_config(
                project_dir,
                config,
                run_file_path=run_file,
                log_callback=print,
                progress_callback=_print_progress,
            )
            _print_summary(summary)
        except Exception as exc:
            failures.append((project_dir, str(exc)))
            print(f"FAILED {project_dir}: {exc}")
            if not bool(args.keep_going):
                break
    if failures:
        print("")
        print(
            "Cluster dynamics batch completed with "
            f"{len(failures)} failure(s)."
        )
        return 1
    print("")
    print("Cluster dynamics batch complete")
    return 0


def _handle_parallel_batch_run(
    project_dirs: list[Path],
    *,
    workers: int,
    keep_going: bool,
) -> int:
    failures: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_project = {
            executor.submit(
                _run_project_collecting_logs,
                project_dir,
            ): project_dir
            for project_dir in project_dirs
        }
        for future in as_completed(future_to_project):
            project_dir = future_to_project[future]
            try:
                summary, log_lines = future.result()
            except Exception as exc:
                failures.append((project_dir, str(exc)))
                print(f"FAILED {project_dir}: {exc}")
                if not keep_going:
                    for pending in future_to_project:
                        if pending is not future:
                            pending.cancel()
                    break
                continue
            for line in log_lines:
                print(line)
            _print_summary(summary)
    if failures:
        print("")
        print(
            "Cluster dynamics batch completed with "
            f"{len(failures)} failure(s)."
        )
        return 1
    print("")
    print("Cluster dynamics batch complete")
    return 0


def _run_project_collecting_logs(project_dir: Path):
    log_lines: list[str] = []

    def log(message: str) -> None:
        log_lines.append(f"[{project_dir.name}] {message}")

    def progress(processed: int, total: int, frame_name: str) -> None:
        log_lines.append(
            f"[{project_dir.name}] {processed}/{total} {frame_name}"
        )

    run_file = default_clusterdynamics_run_file_path(project_dir)
    config = load_clusterdynamics_run_config(run_file)
    summary = run_clusterdynamics_run_config(
        project_dir,
        config,
        run_file_path=run_file,
        log_callback=log,
        progress_callback=progress,
    )
    return summary, log_lines


def _resolve_run_file(project_dir: Path, run_file: Path | None) -> Path:
    if run_file is None:
        return default_clusterdynamics_run_file_path(project_dir)
    return Path(run_file).expanduser().resolve()


def _print_progress(processed: int, total: int, frame_name: str) -> None:
    print(f"{processed}/{total} {frame_name}")


def _print_summary(summary) -> None:
    print("")
    print("Cluster dynamics CLI run complete")
    print(f"Frames folder: {summary.frames_dir}")
    print(f"Output dataset: {summary.output_file}")
    print(f"Frames analyzed: {summary.result.analyzed_frames}")
    print(f"Time bins: {summary.result.bin_count}")
    print(f"Cluster labels: {len(summary.result.cluster_labels)}")
    print(f"Lifetime rows: {len(summary.result.lifetime_by_label)}")
    print(f"Files written: {summary.written_count}")
    if summary.project_file is not None:
        print(f"Project file: {summary.project_file}")


__all__ = ["build_parser", "main"]
