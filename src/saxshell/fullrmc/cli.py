from __future__ import annotations

import argparse
from pathlib import Path

from saxshell.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fullrmc",
        description=(
            "Launch the RMC setup Qt UI or inspect future fullrmc setup "
            "workflows. Running without a subcommand launches the UI."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the fullrmc tool version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")
    ui_parser = subparsers.add_parser("ui", help="Launch the rmcsetup Qt UI.")
    ui_parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        help="Optional SAXS project directory to prefill in the UI.",
    )
    ui_parser.set_defaults(handler=_handle_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"fullrmc {__version__}")
        return 0

    if args.command is None:
        return _handle_ui(args)

    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _handle_ui(args: argparse.Namespace) -> int:
    from .ui.main_window import launch_rmcsetup_ui

    return launch_rmcsetup_ui(getattr(args, "project_dir", None))


__all__ = ["build_parser", "main"]
