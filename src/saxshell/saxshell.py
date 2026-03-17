import argparse

from saxshell.version import __version__  # noqa


def main():
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
        action="store_true",
        help="Show the program's version number and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(f"saxshell {__version__}")
    else:
        # Default behavior when no arguments are given
        parser.print_help()


if __name__ == "__main__":
    main()
