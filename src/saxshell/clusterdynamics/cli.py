from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clusterdynamics",
        description=(
            "Analyze time-binned cluster distributions from extracted PDB "
            "or XYZ frame folders, or launch the Qt UI. Running without "
            "additional arguments launches the UI."
        ),
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
    args = parser.parse_args(argv)

    from .ui.main_window import launch_clusterdynamics_ui

    return launch_clusterdynamics_ui(
        getattr(args, "frames_dir", None),
        energy_file=getattr(args, "energy_file", None),
        project_dir=getattr(args, "project_dir", None),
    )


__all__ = ["main"]
