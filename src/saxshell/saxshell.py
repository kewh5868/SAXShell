from __future__ import annotations

import argparse

from saxshell.version import __version__  # noqa


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saxshell",
        description=(
            "Python package for analysis of small-angle scattering data "
            "from molecular dynamics derived liquid structures.\n\n"
            "For more information, visit: "
            "https://github.com/kewh5868/saxshell/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"saxshell {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    mdtrajectory_parser = subparsers.add_parser(
        "mdtrajectory",
        help="Inspect trajectories, suggest cutoffs, and export frames.",
    )
    mdtrajectory_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the mdtrajectory command.",
    )

    args = parser.parse_args(argv)

    if args.command == "mdtrajectory":
        from saxshell.mdtrajectory.cli import main as mdtrajectory_main

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return mdtrajectory_main(forwarded_args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
