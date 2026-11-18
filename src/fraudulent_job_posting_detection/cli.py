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
    subparsers.add_parser("preprocess", help="stage 2: clean text and persist the corpus")
    subparsers.add_parser("split", help="stage 2: build and audit the leakage-controlled split")
    compare_parser = subparsers.add_parser(
        "compare", help="stage 4: cross-validate candidate models (resumable)"
    )
    compare_parser.add_argument("--only", help="run a single model by name")
    subparsers.add_parser("imbalance", help="stage 4b: compare imbalance strategies")
    subparsers.add_parser("evaluate", help="stage 5: held-out evaluation and interpretability")
    serve_parser = subparsers.add_parser("serve", help="launch the web interface")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5000)
    serve_parser.add_argument("--debug", action="store_true")

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

    if args.command == "compare":
        from .models import run_comparison

        print(run_comparison(only=args.only).to_string(index=False))
        return 0

    if args.command == "imbalance":
        from .models import run_imbalance_comparison

        print(run_imbalance_comparison().to_string(index=False))
        return 0

    if args.command == "serve":
        from .webapp.app import create_app

        app = create_app()
        print(f"\n  Fraud detection UI -> http://{args.host}:{args.port}\n")
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0

    if args.command == "evaluate":
        from .evaluate import run_evaluation

        print(run_evaluation().to_string(index=False))
        return 0

    if args.command == "preprocess":
        from .preprocess import run_preprocess

        run_preprocess()
        return 0

    if args.command == "split":
        import pandas as pd

        from . import config
        from .splits import audit_split, train_test_indices

        corpus = config.PROCESSED_DIR / "corpus.parquet"
        if not corpus.exists():
            print("Run 'fjpd preprocess' first.", file=sys.stderr)
            return 1
        frame = pd.read_parquet(corpus)
        train_idx, test_idx = train_test_indices(frame)
        print(audit_split(frame, train_idx, test_idx).to_string(index=False))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
