"""Command-line entry point. Each pipeline stage is a subcommand.

Running stages through one CLI keeps the project reproducible: the report can
cite the exact command that produced each figure and table.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fjpd",
        description="Fraudulent job posting detection pipeline.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="verify the raw dataset is present and valid")
    subparsers.add_parser("explore", help="stage 1: exploratory data analysis")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "check":
        from .data import DatasetError, class_balance, load_raw

        try:
            frame = load_raw()
        except DatasetError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Dataset OK: {len(frame):,} rows x {frame.shape[1]} columns\n")
        print(class_balance(frame).to_string(index=False))
        return 0

    if args.command == "explore":
        from .explore import run_eda

        run_eda()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
