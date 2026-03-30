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
            "https://github.com/kewh5868/SAXSShell/"
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
    cluster_parser = subparsers.add_parser(
        "cluster",
        help=(
            "Inspect or analyze extracted frame folders, or launch the "
            "cluster UI."
        ),
    )
    cluster_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the cluster UI command.",
    )
    bondanalysis_parser = subparsers.add_parser(
        "bondanalysis",
        help=(
            "Run bond-pair and angle-distribution analysis, or launch the "
            "bondanalysis UI."
        ),
    )
    bondanalysis_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the bondanalysis command.",
    )
    clusterdynamics_parser = subparsers.add_parser(
        "clusterdynamics",
        help=(
            "Analyze time-binned cluster distributions and lifetimes, or "
            "launch the cluster-dynamics UI."
        ),
    )
    clusterdynamics_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the clusterdynamics command.",
    )
    clusterdynamicsml_parser = subparsers.add_parser(
        "clusterdynamicsml",
        help=(
            "Predict larger-cluster surrogate structures, stoichiometries, "
            "and cluster-only SAXS traces."
        ),
    )
    clusterdynamicsml_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the clusterdynamicsml command.",
    )
    xyz2pdb_parser = subparsers.add_parser(
        "xyz2pdb",
        help=(
            "Convert XYZ files to PDB using reference molecules and "
            "residue-assignment rules."
        ),
    )
    xyz2pdb_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the xyz2pdb command.",
    )
    saxs_parser = subparsers.add_parser(
        "saxs",
        help=(
            "Build SAXS fitting projects, refine prefit models, and "
            "launch the SAXS fitting UI."
        ),
    )
    saxs_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the SAXS command.",
    )
    fullrmc_parser = subparsers.add_parser(
        "fullrmc",
        help=(
            "Launch the rmcsetup fullrmc-preparation UI or future "
            "fullrmc setup helpers."
        ),
    )
    fullrmc_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the fullrmc command.",
    )

    args = parser.parse_args(argv)

    if args.command == "mdtrajectory":
        from saxshell.mdtrajectory.cli import main as mdtrajectory_main

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return mdtrajectory_main(forwarded_args)

    if args.command == "cluster":
        from saxshell.cluster.cli import main as cluster_main

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return cluster_main(forwarded_args)

    if args.command == "bondanalysis":
        from saxshell.bondanalysis.cli import main as bondanalysis_main

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return bondanalysis_main(forwarded_args)

    if args.command == "clusterdynamics":
        from saxshell.clusterdynamics.cli import main as clusterdynamics_main

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return clusterdynamics_main(forwarded_args)

    if args.command == "clusterdynamicsml":
        from saxshell.clusterdynamicsml.cli import (
            main as clusterdynamicsml_main,
        )

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return clusterdynamicsml_main(forwarded_args)

    if args.command == "xyz2pdb":
        from saxshell.xyz2pdb.cli import main as xyz2pdb_main

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return xyz2pdb_main(forwarded_args)

    if args.command == "saxs":
        from saxshell.saxs.cli import main as saxs_main

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return saxs_main(forwarded_args)

    if args.command == "fullrmc":
        from saxshell.fullrmc.cli import main as fullrmc_main

        forwarded_args = list(args.args)
        if forwarded_args[:1] == ["--"]:
            forwarded_args = forwarded_args[1:]
        return fullrmc_main(forwarded_args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
