"""The deeper analyses, as jobs the interface can run.

These three stages used to exist only as command-line subcommands, which
meant the interface could not produce the numbers that belong in a report
and the project had two surfaces to keep in step. They are orchestration
only: the statistics live in `explore.py`, `models.py` and `evaluate.py`, and
are called here exactly as the CLI called them.

They are slow by nature -- cross-validation fits every model five times --
so each one reports progress and each one is cancellable at its step
boundaries.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from . import config
from .progress import ProgressReporter

logger = logging.getLogger(__name__)

# Where the last run of each stage is kept, so results survive a restart.
ANALYSIS_PATH = config.MODELS_DIR / "ui_analysis.json"

# Figures these stages write. Declared so the interface can show exactly what
# exists rather than guessing at filenames.
EDA_FIGURE_NAMES: tuple[str, ...] = (
    "text_length_description.png",
    "fraud_rate_binary_flags.png",
    "distinguishing_terms.png",
)
EVALUATION_FIGURE_NAMES: tuple[str, ...] = (
    "precision_recall_test.png",
    "threshold_tradeoff.png",
    "confusion_matrix.png",
    "top_features.png",
    "embedding_training.png",
    "embedding_neighbours.png",
)
ANALYSIS_FIGURES: tuple[str, ...] = EDA_FIGURE_NAMES + EVALUATION_FIGURE_NAMES


# ---------------------------------------------------------------------------
# Stage descriptions, for the interface to build its controls from
# ---------------------------------------------------------------------------
STAGE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "report_eda",
        "label": "Exploratory tables and figures",
        "blurb": (
            "Writes every stage-1 table to reports/tables as CSV and Markdown, "
            "plus the three report figures. The Explore charts are for reading "
            "on screen; this is what goes in the write-up."
        ),
        "cost": "about 20 seconds",
        "controls": [],
    },
    {
        "key": "report_compare",
        "label": "Cross-validated model comparison",
        "blurb": (
            "Grouped, stratified k-fold over the training split for every "
            "selected model. These are the numbers to quote: the single split "
            "on the Train stage is for responsiveness, not for a report. "
            "Resumable — each finished model is written to disk immediately."
        ),
        "cost": "minutes per model",
        "controls": ["models", "folds", "feature_sets", "restart"],
    },
    {
        "key": "report_imbalance",
        "label": "Imbalance strategy comparison",
        "blurb": (
            "No treatment against class weighting against SMOTE, on LinearSVC. "
            "Shows that the treatment moves the operating point rather than "
            "the model's ability to rank."
        ),
        "cost": "about 2 minutes",
        "controls": ["folds"],
    },
    {
        "key": "report_evaluate",
        "label": "Final held-out evaluation",
        "blurb": (
            "The test split is touched once, here: the strongest "
            "configurations scored at both their default and tuned "
            "thresholds, with coefficients and diagnostic figures."
        ),
        "cost": "several minutes",
        "controls": [],
    },
)

STAGE_KEYS: tuple[str, ...] = tuple(spec["key"] for spec in STAGE_SPECS)


def catalogue() -> list[dict[str, Any]]:
    return [dict(spec) for spec in STAGE_SPECS]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_analysis() -> dict[str, Any]:
    if not ANALYSIS_PATH.exists():
        return {}
    try:
        payload = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("could not read %s", ANALYSIS_PATH.name)
        return {}


def _store(key: str, payload: dict[str, Any]) -> None:
    """Merge one stage's result into the stored analysis."""
    config.ensure_directories()
    current = load_analysis()
    current[key] = payload
    ANALYSIS_PATH.write_text(
        json.dumps(current, indent=2, default=str), encoding="utf-8"
    )


def _table_payload(table: pd.DataFrame) -> dict[str, Any]:
    """A DataFrame as JSON the page can render directly."""
    return {
        "columns": [str(column) for column in table.columns],
        "rows": json.loads(table.to_json(orient="values", default_handler=str)),
    }


def _timestamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def run_report_eda(progress: ProgressReporter | None = None) -> dict[str, Any]:
    from .explore import run_eda

    reporter = progress if progress is not None else ProgressReporter()
    run_eda(progress=reporter)
    payload = {
        "ran_at": _timestamp(),
        "tables": _written_tables(),
        "figures": list(EDA_FIGURE_NAMES),
    }
    _store("report_eda", payload)
    return payload


def run_report_compare(
    models: list[str] | None = None,
    folds: int = 5,
    feature_sets: list[str] | None = None,
    restart: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Cross-validate each selected model, resuming unless asked to restart."""
    from .models import (
        FEATURE_SETS,
        REGISTRY,
        RESULTS_PATH,
        available_models,
        cross_validate_models,
        read_corpus,
        validate_model_names,
    )
    from .plotting import save_table
    from .splits import train_test_indices

    reporter = progress if progress is not None else ProgressReporter()
    config.ensure_directories()

    names = validate_model_names(models or list(available_models()))
    if not names:
        raise ValueError("Select at least one model.")
    folds = max(2, min(int(folds), 10))

    chosen_sets = {
        label: flag
        for label, flag in FEATURE_SETS.items()
        if not feature_sets or label in set(feature_sets)
    } or dict(FEATURE_SETS)

    if restart:
        RESULTS_PATH.unlink(missing_ok=True)

    done: set[tuple[str, str]] = set()
    if RESULTS_PATH.exists():
        try:
            existing = pd.read_csv(RESULTS_PATH)
            if {"model", "features"} <= set(existing.columns):
                done = set(zip(existing["model"], existing["features"], strict=True))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            RESULTS_PATH.unlink(missing_ok=True)

    reporter(0.01, "loading the cleaned corpus")
    frame = read_corpus(copy=False)
    train_idx, _ = train_test_indices(frame, n_splits=5)

    jobs = [
        (label, flag, name)
        for label, flag in chosen_sets.items()
        for name in names
    ]
    # Weight by the model's relative cost so the bar tracks time, not count.
    weights = {
        f"{label}|{name}": REGISTRY[name].cost for label, _flag, name in jobs
    }
    spans = reporter.span(0.97, "cross-validating").weighted(weights)

    for label, flag, name in jobs:
        step = spans[f"{label}|{name}"]
        if (name, label) in done:
            step.done(f"{REGISTRY[name].label} / {label} already done")
            continue
        step(0.05, f"{REGISTRY[name].label} / {label} ({folds} folds)")
        result = cross_validate_models(
            frame, train_idx, [name], use_structured=flag, n_splits=folds
        )
        result["features"] = label
        result.to_csv(
            RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False
        )
        step.done(f"{REGISTRY[name].label} / {label} done")

    table = pd.read_csv(RESULTS_PATH)
    ordered = [
        "model", "features", "f1_fraud", "f1_fraud_sd", "precision_fraud",
        "recall_fraud", "avg_precision", "roc_auc", "fit_seconds",
    ]
    table = table[[c for c in ordered if c in table.columns]]
    table = table.sort_values("f1_fraud", ascending=False).reset_index(drop=True)
    save_table(table, "model_comparison_cv")

    payload = {
        "ran_at": _timestamp(),
        "folds": folds,
        "models": names,
        "feature_sets": list(chosen_sets),
        "table": _table_payload(table),
    }
    _store("report_compare", payload)
    reporter.done("comparison complete")
    return payload


def run_report_imbalance(
    folds: int = 5, progress: ProgressReporter | None = None
) -> dict[str, Any]:
    from .models import run_imbalance_comparison

    reporter = progress if progress is not None else ProgressReporter()
    reporter(0.05, "comparing imbalance strategies")
    table = run_imbalance_comparison(n_splits=max(2, min(int(folds), 10)),
                                     progress=reporter)
    payload = {
        "ran_at": _timestamp(),
        "folds": int(folds),
        "table": _table_payload(table),
    }
    _store("report_imbalance", payload)
    reporter.done("imbalance comparison complete")
    return payload


def run_report_evaluate(progress: ProgressReporter | None = None) -> dict[str, Any]:
    from .evaluate import run_evaluation

    reporter = progress if progress is not None else ProgressReporter()
    table = run_evaluation(progress=reporter)
    payload = {
        "ran_at": _timestamp(),
        "table": _table_payload(table),
        "figures": [
            name
            for name in EVALUATION_FIGURE_NAMES
            if (config.FIGURES_DIR / name).exists()
        ],
        "tables": _written_tables(),
    }
    _store("report_evaluate", payload)
    reporter.done("evaluation complete")
    return payload


def _written_tables() -> list[dict[str, Any]]:
    """Report tables currently on disk, newest first."""
    if not config.TABLES_DIR.exists():
        return []
    rows = []
    for path in sorted(config.TABLES_DIR.glob("*.csv")):
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append({
            "name": path.stem,
            "csv": path.name,
            "markdown": f"{path.stem}.md",
            "bytes": stat.st_size,
        })
    return rows


RUNNERS = {
    "report_eda": lambda params, reporter: run_report_eda(progress=reporter),
    "report_compare": lambda params, reporter: run_report_compare(
        models=params.get("models"),
        folds=params.get("folds", 5),
        feature_sets=params.get("feature_sets"),
        restart=bool(params.get("restart")),
        progress=reporter,
    ),
    "report_imbalance": lambda params, reporter: run_report_imbalance(
        folds=params.get("folds", 5), progress=reporter
    ),
    "report_evaluate": lambda params, reporter: run_report_evaluate(reporter),
}
