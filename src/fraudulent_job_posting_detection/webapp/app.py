"""Flask interface for the fraud detection pipeline.

Four pages, one per stage:

    /          import  -- upload or locate the CSV, validate it
    /explore   explore -- class balance, missingness, figures
    /clean     clean   -- preprocessing options, build the corpus
    /train     train   -- pick models and metrics, see the top 3

Every route is thin: it reads the form, calls the library, and renders a
template. All the analysis lives in the sibling modules, so the same code
backs both the CLI and this interface.
"""

from __future__ import annotations

import logging

import pandas as pd
from flask import (
    Flask, flash, jsonify, redirect, render_template,
    request, send_from_directory, url_for,
)

from .. import config
from ..data import DatasetError, class_balance, load_raw, missingness
from ..models import ALL_METRICS, IMBALANCE_STRATEGIES, candidate_models, train_and_evaluate
from ..preprocess import run_preprocess
from ..splits import audit_split, train_test_indices
from . import jobs

logger = logging.getLogger(__name__)

CORPUS_PATH = config.PROCESSED_DIR / "corpus.parquet"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def corpus_exists() -> bool:
    return CORPUS_PATH.exists()


def read_corpus() -> pd.DataFrame:
    return pd.read_parquet(CORPUS_PATH)


def dataset_summary() -> dict[str, object] | None:
    """Row counts and class balance, or None when the CSV is absent or invalid."""
    try:
        frame = load_raw()
    except DatasetError:
        return None
    balance = class_balance(frame)
    return {
        "rows": len(frame),
        "columns": frame.shape[1],
        "balance": balance.to_dict("records"),
        "fraud_pct": float(balance.loc[1, "percent"]),
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "fjpd-local-only"
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    config.ensure_directories()

    # ---------------------------------------------------------------- import
    @app.route("/")
    def index():
        return render_template(
            "index.html",
            summary=dataset_summary(),
            raw_path=config.RAW_CSV,
            corpus_ready=corpus_exists(),
        )

    @app.post("/upload")
    def upload():
        uploaded = request.files.get("csv")
        if uploaded is None or not uploaded.filename:
            flash("No file selected.", "error")
            return redirect(url_for("index"))
        if not uploaded.filename.lower().endswith(".csv"):
            flash("Expected a .csv file.", "error")
            return redirect(url_for("index"))

        uploaded.save(config.RAW_CSV)
        try:
            load_raw()
        except DatasetError as exc:
            flash(f"File saved but failed validation: {exc}", "error")
            return redirect(url_for("index"))
        flash(f"Loaded and validated {config.RAW_CSV.name}.", "success")
        return redirect(url_for("index"))

    # --------------------------------------------------------------- explore
    @app.route("/explore")
    def explore():
        summary = dataset_summary()
        if summary is None:
            flash("Import the dataset first.", "error")
            return redirect(url_for("index"))

        frame = load_raw()
        tables = {
            "Missing values by column": missingness(frame).head(12),
            "Fraud rate by binary flag": _flag_table(frame),
        }
        return render_template(
            "explore.html",
            summary=summary,
            tables={name: _to_rows(table) for name, table in tables.items()},
            preview=_to_rows(_preview(frame)),
            figures=_available_figures(),
            job=jobs.get("explore").as_dict(),
        )

    @app.post("/explore/run")
    def explore_run():
        from ..explore import run_eda

        def task(progress):
            progress(0, 1, "running exploratory analysis")
            run_eda()
            return "ok"

        if not jobs.start("explore", task):
            flash("Exploratory analysis is already running.", "error")
        return redirect(url_for("explore"))

    # ----------------------------------------------------------------- clean
    @app.route("/clean")
    def clean():
        split_table = None
        if corpus_exists():
            frame = read_corpus()
            train_idx, test_idx = train_test_indices(frame)
            split_table = _to_rows(audit_split(frame, train_idx, test_idx))
        return render_template(
            "clean.html",
            corpus_ready=corpus_exists(),
            split_table=split_table,
            sample=_corpus_sample(),
            job=jobs.get("clean").as_dict(),
        )

    @app.post("/clean/run")
    def clean_run():
        options = {
            "remove_stopwords": "remove_stopwords" in request.form,
            "stem": "stem" in request.form,
            "normalise_placeholders": "normalise_placeholders" in request.form,
        }

        def task(progress):
            progress(0, 1, "cleaning and tokenising 17,880 postings")
            run_preprocess(**options)
            return options

        if not jobs.start("clean", task):
            flash("Cleaning is already running.", "error")
        return redirect(url_for("clean"))

    # ----------------------------------------------------------------- train
    @app.route("/train")
    def train():
        job = jobs.get("train")
        results = job.result if job.status == "done" else None
        chosen = (results or {}).get("metrics", list(ALL_METRICS)) if results else None
        return render_template(
            "train.html",
            corpus_ready=corpus_exists(),
            models=list(candidate_models()),
            all_metrics=ALL_METRICS,
            strategies=IMBALANCE_STRATEGIES,
            job=job.as_dict(),
            results=results,
            chosen_metrics=chosen,
        )

    @app.post("/train/run")
    def train_run():
        selected = request.form.getlist("models")
        metrics = request.form.getlist("metrics") or ["f1_fraud"]
        strategy = request.form.get("strategy", "class_weight")
        rank_by = request.form.get("rank_by", "f1_fraud")

        if not corpus_exists():
            flash("Clean the dataset first.", "error")
            return redirect(url_for("clean"))
        if not selected:
            flash("Select at least one model.", "error")
            return redirect(url_for("train"))

        def task(progress):
            frame = read_corpus()
            table = train_and_evaluate(frame, selected, strategy, progress)
            table = table.sort_values(rank_by, ascending=False).reset_index(drop=True)
            return {
                "rows": table.to_dict("records"),
                "metrics": metrics,
                "rank_by": rank_by,
                "strategy": strategy,
                "podium": table.head(3).to_dict("records"),
            }

        if not jobs.start("train", task):
            flash("Training is already running.", "error")
        return redirect(url_for("train"))

    # ------------------------------------------------------------ job status
    @app.get("/job/<name>")
    def job_status(name: str):
        return jsonify(jobs.get(name).as_dict())

    @app.get("/figure/<path:filename>")
    def figure(filename: str):
        return send_from_directory(config.FIGURES_DIR, filename)

    return app


# ---------------------------------------------------------------------------
# Small view helpers
# ---------------------------------------------------------------------------
def _to_rows(table: pd.DataFrame) -> dict[str, object]:
    return {"columns": list(table.columns), "rows": table.values.tolist()}


def _preview(frame: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    columns = ["title", "location", "has_company_logo", "telecommuting", "fraudulent"]
    preview = frame[columns].head(n).copy()
    preview["title"] = preview["title"].astype(str).str.slice(0, 46)
    preview["location"] = preview["location"].astype(str).str.slice(0, 26)
    return preview


def _flag_table(frame: pd.DataFrame) -> pd.DataFrame:
    from ..explore import fraud_rate_by

    return pd.concat(
        [fraud_rate_by(frame, flag) for flag in config.BINARY_FLAGS], ignore_index=True
    )


def _available_figures() -> list[str]:
    if not config.FIGURES_DIR.exists():
        return []
    return sorted(path.name for path in config.FIGURES_DIR.glob("*.png"))


def _corpus_sample(n: int = 3) -> list[dict[str, str]]:
    if not corpus_exists():
        return []
    frame = read_corpus()
    sample = frame.sample(n, random_state=config.RANDOM_STATE)
    return [
        {
            "label": "fraudulent" if row[config.TARGET] else "legitimate",
            "text": str(row["text_all"])[:300],
        }
        for _, row in sample.iterrows()
    ]
