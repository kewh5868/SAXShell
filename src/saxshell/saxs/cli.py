from __future__ import annotations

import argparse
from pathlib import Path

from saxshell.version import __version__

from ._model_templates import list_template_specs
from .template_installation import (
    format_validation_report,
    install_template_candidate,
    validate_template_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saxs",
        description=(
            "Set up SAXS projects, launch the SAXS Qt UI, inspect bundled "
            "model templates, and manage prefit/DREAM workflows."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the SAXS application version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the SAXS Qt UI.")
    ui_parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        help="Optional project directory to open on launch.",
    )
    ui_parser.set_defaults(handler=_handle_ui)

    templates_parser = subparsers.add_parser(
        "templates",
        help="List, validate, or install SAXS model templates.",
    )
    templates_parser.set_defaults(handler=_handle_templates)
    template_subparsers = templates_parser.add_subparsers(
        dest="templates_command"
    )

    validate_parser = template_subparsers.add_parser(
        "validate",
        help="Validate a candidate SAXS template for prefit and DREAM use.",
    )
    validate_parser.add_argument(
        "template_path",
        type=Path,
        help="Path to the candidate template Python file.",
    )
    validate_parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional paired metadata JSON file for display text/tooltips.",
    )
    validate_parser.set_defaults(handler=_handle_templates_validate)

    install_parser = template_subparsers.add_parser(
        "install",
        help="Validate and install a SAXS template into a template directory.",
    )
    install_parser.add_argument(
        "template_path",
        type=Path,
        help="Path to the candidate template Python file.",
    )
    install_parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional paired metadata JSON file for display text/tooltips.",
    )
    install_parser.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="Destination template directory. Defaults to the bundled template folder.",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing installed template with the same name.",
    )
    install_parser.set_defaults(handler=_handle_templates_install)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"saxs {__version__}")
        return 0

    if args.command is None:
        return _handle_ui(args)

    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _handle_ui(args: argparse.Namespace) -> int:
    from .ui.main_window import launch_saxs_ui

    return launch_saxs_ui(getattr(args, "project_dir", None))


def _handle_templates(_args: argparse.Namespace) -> int:
    for template in list_template_specs():
        print(f"{template.display_name} ({template.name})")
    return 0


def _handle_templates_validate(args: argparse.Namespace) -> int:
    result = validate_template_candidate(
        args.template_path,
        metadata_path=args.metadata,
    )
    print(format_validation_report(result))
    return 0 if result.passed else 1


def _handle_templates_install(args: argparse.Namespace) -> int:
    installed = install_template_candidate(
        args.template_path,
        metadata_path=args.metadata,
        destination_dir=args.template_dir,
        overwrite=bool(args.force),
    )
    print(format_validation_report(installed.validation_result))
    print(f"Installed template: {installed.installed_template_path}")
    if installed.installed_metadata_path is not None:
        print(f"Installed metadata: {installed.installed_metadata_path}")
    return 0


__all__ = ["build_parser", "main"]
