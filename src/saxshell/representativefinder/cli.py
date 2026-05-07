from __future__ import annotations

import argparse
from pathlib import Path

from saxshell.version import __version__

from .run_config import (
    default_representativefinder_run_file_path,
    load_representativefinder_run_config,
    representativefinder_run_targets,
    run_representativefinder_run_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="representativefinder",
        description=(
            "Build or run project-backed representative-structure analysis "
            "run files. Running without a subcommand launches the beta run "
            "file setup UI."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the representativefinder version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    setup_ui_parser = subparsers.add_parser(
        "setup-ui",
        help="Launch the beta Qt run-file setup interface.",
    )
    setup_ui_parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        help="Optional SAXSShell project folder.",
    )
    setup_ui_parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional cluster/stoichiometry folder to prefill.",
    )
    setup_ui_parser.set_defaults(handler=_handle_setup_ui)

    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the full representative-structure analysis UI.",
    )
    ui_parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        help="Optional SAXSShell project folder.",
    )
    ui_parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional cluster/stoichiometry folder to prefill.",
    )
    ui_parser.set_defaults(handler=_handle_ui)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect the targets described by a representative run file.",
    )
    inspect_parser.add_argument(
        "project_dir",
        type=Path,
        help="SAXSShell project folder containing the run file.",
    )
    inspect_parser.add_argument(
        "--run-file",
        type=Path,
        default=None,
        help=(
            "Run file path. Defaults to "
            "representative_structure_cli_run.json in the project folder."
        ),
    )
    inspect_parser.set_defaults(handler=_handle_inspect)

    run_parser = subparsers.add_parser(
        "run",
        help="Run representative-structure analysis from a project run file.",
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
            "Run file path. Defaults to "
            "representative_structure_cli_run.json in the project folder."
        ),
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker thread count stored in the run file.",
    )
    run_parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recalculate stoichiometries that already have saved project representatives.",
    )
    run_parser.set_defaults(handler=_handle_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"representativefinder {__version__}")
        return 0

    if args.command is None:
        return _handle_setup_ui(args)

    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _handle_setup_ui(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.run_file_window import launch_representativefinder_run_file_ui

    owns_app = QApplication.instance() is None
    launch_representativefinder_run_file_ui(
        initial_project_dir=getattr(args, "project_dir", None),
        initial_input_path=getattr(args, "input_dir", None),
    )
    app = QApplication.instance()
    if owns_app and app is not None:
        return app.exec()
    return 0


def _handle_ui(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import launch_representativefinder_ui

    owns_app = QApplication.instance() is None
    launch_representativefinder_ui(
        initial_project_dir=getattr(args, "project_dir", None),
        initial_input_path=getattr(args, "input_dir", None),
    )
    app = QApplication.instance()
    if owns_app and app is not None:
        return app.exec()
    return 0


def _handle_inspect(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).expanduser().resolve()
    run_file = _resolve_run_file(project_dir, args.run_file)
    config = load_representativefinder_run_config(run_file)
    targets, skipped_existing = representativefinder_run_targets(
        project_dir=project_dir,
        config=config,
    )
    print(f"Project folder: {project_dir}")
    print(f"Run file: {run_file}")
    print(f"Analysis mode: {config.analysis_mode}")
    print(f"Targets to run: {len(targets)}")
    if skipped_existing:
        print("Skipped existing: " + ", ".join(skipped_existing))
    for target in targets:
        print(
            f"- {target.inspection.structure_label}: "
            f"{target.inspection.candidate_count} candidate file(s) -> "
            f"{target.output_dir}"
        )
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).expanduser().resolve()
    run_file = _resolve_run_file(project_dir, args.run_file)
    config = load_representativefinder_run_config(run_file)
    if args.workers is not None:
        config.settings = config.settings.__class__(
            selection_algorithm=config.settings.selection_algorithm,
            bond_weight=config.settings.bond_weight,
            angle_weight=config.settings.angle_weight,
            solvent_weight=config.settings.solvent_weight,
            generate_predicted_optimized_representative=(
                config.settings.generate_predicted_optimized_representative
            ),
            parallel_workers=int(args.workers),
            quantiles=config.settings.quantiles,
            bond_pairs=config.settings.bond_pairs,
            angle_triplets=config.settings.angle_triplets,
        )
    if bool(args.overwrite_existing):
        config.overwrite_existing = True

    summary = run_representativefinder_run_config(
        project_dir,
        config,
        run_file_path=run_file,
        log_callback=print,
        progress_callback=_print_progress,
    )
    print("")
    print("Representative CLI run complete")
    print(f"Completed: {summary.completed_count}")
    print(f"Failed: {summary.failed_count}")
    if summary.skipped_existing:
        print("Skipped existing: " + ", ".join(summary.skipped_existing))
    for path in summary.project_representative_paths:
        print(f"Project representative: {path}")
    for failure in summary.failures:
        print(f"FAILED {failure.structure_label}: {failure.message}")
    return 1 if summary.failures else 0


def _resolve_run_file(project_dir: Path, run_file: Path | None) -> Path:
    if run_file is None:
        return default_representativefinder_run_file_path(project_dir)
    return Path(run_file).expanduser().resolve()


def _print_progress(processed: int, total: int, message: str) -> None:
    print(f"{processed}/{total} {message}")


__all__ = ["build_parser", "main"]
