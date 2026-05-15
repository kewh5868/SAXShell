from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from saxshell.version import __version__

from .run_config import (
    default_clusterdynamicsml_run_file_path,
    load_clusterdynamicsml_run_config,
    run_clusterdynamicsml_run_config,
)

_COMMANDS = {"setup-ui", "ui", "run", "batch-run"}
_TOP_LEVEL_OPTIONS = {"-h", "--help", "--version"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clusterdynamicsml",
        description=(
            "Predict larger-cluster stoichiometries, representative "
            "structures, and cluster-only SAXS traces. Running without a "
            "subcommand launches the Qt UI."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the clusterdynamicsml version number and exit.",
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
    setup_ui_parser.add_argument(
        "--clusters-dir",
        type=Path,
        default=None,
        help="Optional smaller-cluster structure directory to prefill.",
    )
    setup_ui_parser.add_argument(
        "--experimental-data",
        type=Path,
        default=None,
        help="Optional experimental SAXS data file to prefill.",
    )
    setup_ui_parser.set_defaults(handler=_handle_setup_ui)

    ui_parser = subparsers.add_parser("ui", help="Launch the Qt UI.")
    _add_ui_prefill_arguments(ui_parser)
    ui_parser.set_defaults(handler=_handle_ui)

    run_parser = subparsers.add_parser(
        "run",
        help="Run cluster dynamics ML from a project-backed run file.",
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
            "Run file path. Defaults to cluster_dynamics_ml_cli_run.json "
            "in the project folder."
        ),
    )
    run_parser.set_defaults(handler=_handle_run)

    batch_parser = subparsers.add_parser(
        "batch-run",
        help=(
            "Run the default cluster dynamics ML run file for multiple "
            "projects."
        ),
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
        print(f"clusterdynamicsml {__version__}")
        return 0
    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _add_ui_prefill_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "frames_dir",
        nargs="?",
        type=Path,
        help="Optional extracted frames directory to prefill in the UI.",
    )
    parser.add_argument(
        "--energy-file",
        type=Path,
        default=None,
        help="Optional CP2K .ener file to prefill in the UI.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Optional SAXSShell project directory to prefill in the UI.",
    )
    parser.add_argument(
        "--clusters-dir",
        type=Path,
        default=None,
        help="Optional smaller-cluster structure directory to prefill in the UI.",
    )
    parser.add_argument(
        "--experimental-data",
        type=Path,
        default=None,
        help="Optional experimental SAXS data file to prefill in the UI.",
    )


def _handle_legacy_ui(raw_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="clusterdynamicsml",
        description="Launch the SAXSShell clusterdynamicsml UI.",
    )
    _add_ui_prefill_arguments(parser)
    args = parser.parse_args(raw_args)
    return _launch_ui_from_args(args)


def _handle_setup_ui(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.run_file_window import launch_clusterdynamicsml_run_file_ui

    owns_app = QApplication.instance() is None
    launch_clusterdynamicsml_run_file_ui(
        initial_project_dir=getattr(args, "project_dir", None),
        initial_frames_dir=getattr(args, "frames_dir", None),
        initial_energy_file=getattr(args, "energy_file", None),
        initial_clusters_dir=getattr(args, "clusters_dir", None),
        initial_experimental_data_file=getattr(
            args, "experimental_data", None
        ),
    )
    app = QApplication.instance()
    if owns_app and app is not None:
        return app.exec()
    return 0


def _handle_ui(args: argparse.Namespace) -> int:
    return _launch_ui_from_args(args)


def _launch_ui_from_args(args: argparse.Namespace) -> int:
    from .ui.main_window import launch_clusterdynamicsml_ui

    return launch_clusterdynamicsml_ui(
        getattr(args, "frames_dir", None),
        energy_file=getattr(args, "energy_file", None),
        project_dir=getattr(args, "project_dir", None),
        clusters_dir=getattr(args, "clusters_dir", None),
        experimental_data_file=getattr(args, "experimental_data", None),
    )


def _handle_run(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).expanduser().resolve()
    run_file = _resolve_run_file(project_dir, args.run_file)
    config = load_clusterdynamicsml_run_config(run_file)
    summary = run_clusterdynamicsml_run_config(
        project_dir,
        config,
        run_file_path=run_file,
        log_callback=print,
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
            run_file = default_clusterdynamicsml_run_file_path(project_dir)
            config = load_clusterdynamicsml_run_config(run_file)
            summary = run_clusterdynamicsml_run_config(
                project_dir,
                config,
                run_file_path=run_file,
                log_callback=print,
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
            "Cluster dynamics ML batch completed with "
            f"{len(failures)} failure(s)."
        )
        return 1
    print("")
    print("Cluster dynamics ML batch complete")
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
            "Cluster dynamics ML batch completed with "
            f"{len(failures)} failure(s)."
        )
        return 1
    print("")
    print("Cluster dynamics ML batch complete")
    return 0


def _run_project_collecting_logs(project_dir: Path):
    log_lines: list[str] = []

    def log(message: str) -> None:
        log_lines.append(f"[{project_dir.name}] {message}")

    run_file = default_clusterdynamicsml_run_file_path(project_dir)
    config = load_clusterdynamicsml_run_config(run_file)
    summary = run_clusterdynamicsml_run_config(
        project_dir,
        config,
        run_file_path=run_file,
        log_callback=log,
    )
    return summary, log_lines


def _resolve_run_file(project_dir: Path, run_file: Path | None) -> Path:
    if run_file is None:
        return default_clusterdynamicsml_run_file_path(project_dir)
    return Path(run_file).expanduser().resolve()


def _print_summary(summary) -> None:
    print("")
    print("Cluster dynamics ML CLI run complete")
    print(f"Frames folder: {summary.frames_dir}")
    print(f"Output dataset: {summary.output_file}")
    print(f"Frames analyzed: {summary.result.dynamics_result.analyzed_frames}")
    print(f"Time bins: {summary.result.dynamics_result.bin_count}")
    print(
        f"Training observations: {len(summary.result.training_observations)}"
    )
    print(f"Predictions: {len(summary.result.predictions)}")
    print(f"Files written: {summary.written_count}")
    if summary.project_file is not None:
        print(f"Project file: {summary.project_file}")


__all__ = ["build_parser", "main"]
