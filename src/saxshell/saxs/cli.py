from __future__ import annotations

import argparse
from pathlib import Path

from saxshell.version import __version__

from ._model_templates import list_template_specs


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
        help="List the bundled SAXS model templates.",
    )
    templates_parser.set_defaults(handler=_handle_templates)
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


__all__ = ["build_parser", "main"]
