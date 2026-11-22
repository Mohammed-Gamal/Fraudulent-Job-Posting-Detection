"""Command-line entry point.

This used to carry a subcommand per pipeline stage, which meant the project
had two surfaces doing the same work: a web interface that could clean and
train, and a CLI that could additionally cross-validate and evaluate. The
two drifted, and the numbers that belong in a report were only reachable from
the terminal.

Every stage now lives in the interface, including the slow report analyses,
so this file does one job: start it. `check` is kept because confirming the
dataset parses is useful before the browser is even open.
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
    # matplotlib's font manager logs a line per glyph lookup at DEBUG.
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fjpd",
        description=(
            "Fraudulent job posting detection. Every pipeline stage runs in "
            "the web interface: start it with `fjpd serve`."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve", help="launch the web interface (every stage lives there)"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5000)
    serve_parser.add_argument("--debug", action="store_true")

    subparsers.add_parser("check", help="verify the raw dataset is present and valid")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    # No subcommand is the common case: start the interface.
    command = args.command or "serve"

    if command == "check":
        from .data import DatasetError, class_balance, load_raw

        try:
            frame = load_raw()
        except DatasetError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Dataset OK: {len(frame):,} rows x {frame.shape[1]} columns\n")
        print(class_balance(frame).to_string(index=False))
        return 0

    if command == "serve":
        from .webapp.app import create_app

        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 5000)
        app = create_app()
        print(f"\n  Fraud detection interface -> http://{host}:{port}\n")
        # threaded: the page polls job progress while a worker thread runs,
        # and fetches each chart separately. Single-threaded, a running job
        # would block every poll and the bar would never move.
        app.run(
            host=host,
            port=port,
            debug=getattr(args, "debug", False),
            threaded=True,
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
