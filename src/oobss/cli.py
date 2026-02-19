from __future__ import annotations

import argparse

from .separators import __all__ as separator_exports


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="oobss",
        description="Open online blind source separation toolkit",
    )
    parser.add_argument(
        "--list-separators",
        action="store_true",
        help="Print available separator names and exit",
    )
    args = parser.parse_args()

    if args.list_separators:
        names = sorted(name for name in separator_exports if name[0].isupper())
        for name in names:
            print(name)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
