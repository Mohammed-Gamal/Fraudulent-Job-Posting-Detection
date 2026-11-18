"""Flask interface for the fraud detection pipeline.

Five pages, each owning one stage and showing only that stage's output:

    /          import  -- upload the CSV and validate it
    /explore   explore -- raw distributions and the stage-1 figures
    /clean     clean   -- cleaning options, corpus provenance, cleaned samples
    /train     train   -- the split, model selection, results and diagnostics
    /predict   predict -- score a pasted advert with the saved model

Routes stay thin: read the form, call the library, render a template. All
analysis lives in the sibling modules, so the CLI and this interface share
one implementation and cannot drift apart.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from flask import (
    Flask, flash, jsonify, redirect, render_template,
    request, send_from_directory, url_for,
)

from .. import config
from ..data import DatasetError, class_balance, load_raw, missingness
from ..explore import EDA_FIGURES
from ..models import (
    ALL_METRICS, IMBALANCE_STRATEGIES, UI_FIGURES,
    candidate_models, load_ui_model, train_and_evaluate,
)
from ..predict import corpus_options, predict_posting
from ..preprocess import CORPUS_PATH, DEFAULT_OPTIONS, read_corpus_meta, run_preprocess
from ..splits import audit_split, train_test_indices
from . import jobs

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Training results are written here as well as held in memory. The best model
# is already persisted, so without this a restart would leave a saved model
# with no results beside it -- and the diagnostic plots on disk would never be
# shown again.
RESULTS_PATH = config.MODELS_DIR / "ui_results.json"

OPTION_LABELS: dict[str, str] = {
    "normalise_placeholders": "Normalise redaction placeholders",
    "remove_stopwords": "Remove stopwords",
    "stem": "Porter stemming",
}


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


def save_results(payload: dict[str, object]) -> None:
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_results() -> dict[str, object] | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def corpus_provenance() -> dict[str, object] | None:
    """What built the stored corpus, formatted for display."""
    meta = read_corpus_meta()
    if meta is None:
        return None
    options = {**DEFAULT_OPTIONS, **meta.get("options", {})}
    return {
        "built_at": meta.get("built_at", "unknown"),
        "rows": meta.get("rows"),
        "median_tokens": meta.get("median_tokens"),
        "options": [
            {"label": OPTION_LABELS.get(key, key), "enabled": bool(value)}
            for key, value in options.items()
        ],
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
            "index.html", summary=dataset_summary(), raw_path=config.RAW_CSV
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
            figures=_existing(EDA_FIGURES),
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
        return render_template(
            "clean.html",
            corpus_ready=corpus_exists(),
            provenance=corpus_provenance(),
            sample=_corpus_sample(),
            job=jobs.get("clean").as_dict(),
        )

    @app.post("/clean/run")
    def clean_run():
        options = {key: key in request.form for key in DEFAULT_OPTIONS}

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
        if not corpus_exists():
            return render_template("train.html", corpus_ready=False)

        frame = read_corpus()
        train_idx, test_idx = train_test_indices(frame)
        job = jobs.get("train")
        return render_template(
            "train.html",
            corpus_ready=True,
            provenance=corpus_provenance(),
            split_table=_to_rows(audit_split(frame, train_idx, test_idx)),
            models=list(candidate_models()),
            all_metrics=ALL_METRICS,
            strategies=IMBALANCE_STRATEGIES,
            job=job.as_dict(),
            # Prefer this session's result; fall back to the last run on disk.
            results=job.result if job.status == "done" else load_results(),
            figures=_existing(UI_FIGURES),
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
            table = train_and_evaluate(
                read_corpus(), selected, strategy, rank_by, progress
            )
            payload = {
                "rows": table.to_dict("records"),
                "metrics": metrics,
                "rank_by": rank_by,
                "strategy": strategy,
                "podium": table.head(3).to_dict("records"),
            }
            save_results(payload)
            return payload

        if not jobs.start("train", task):
            flash("Training is already running.", "error")
        return redirect(url_for("train"))

    # --------------------------------------------------------------- predict
    @app.route("/predict")
    def predict():
        saved = load_ui_model()
        return render_template(
            "predict.html",
            model_name=saved[0] if saved else None,
            options=corpus_options(),
            example=_EXAMPLE_POSTING,
            form={},
            result=None,
        )

    @app.post("/predict")
    def predict_run():
        saved = load_ui_model()
        if saved is None:
            flash("Train a model first — the prediction page uses the best one.", "error")
            return redirect(url_for("train"))

        name, pipeline = saved
        values: dict[str, object] = {
            field: request.form.get(field, "") for field in config.TEXT_FIELDS
        }
        for flag in config.BINARY_FLAGS:
            values[flag] = 1 if flag in request.form else 0

        if not str(values.get("description", "")).strip():
            flash("A description is required.", "error")
            return redirect(url_for("predict"))

        try:
            result = predict_posting(pipeline, values)
        except Exception as exc:
            logger.exception("prediction failed")
            flash(f"Prediction failed: {type(exc).__name__}: {exc}", "error")
            return redirect(url_for("predict"))

        return render_template(
            "predict.html",
            model_name=name,
            options=corpus_options(),
            example=_EXAMPLE_POSTING,
            form=request.form,
            result=result,
        )

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
_EXAMPLE_POSTING = {
    "title": "Data Entry Clerk - Work From Home",
    "description": (
        "Earn $5000 per month working from home! No experience necessary. "
        "We are recruiting motivated individuals to start immediately. "
        "Apply using the link below and send your details to our recruiter."
    ),
    "requirements": "Must have a computer and internet access. High school diploma.",
}


def _to_rows(table: pd.DataFrame) -> dict[str, object]:
    return {"columns": list(table.columns), "rows": table.values.tolist()}


def _existing(names: tuple[str, ...]) -> list[str]:
    """Keep only the declared figures that have actually been generated."""
    return [name for name in names if (config.FIGURES_DIR / name).exists()]


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
