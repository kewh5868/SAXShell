from __future__ import annotations

from saxshell.cluster.cli import main as cluster_main


def main(argv: list[str] | None = None) -> int:
    return cluster_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
