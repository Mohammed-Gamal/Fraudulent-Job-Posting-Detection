"""Flask interface for the fraud detection pipeline: one page, one JSON API.

The five-page version made the pipeline look like five unrelated tools and
reloaded the whole document to show a progress bar. This serves a single
document whose five stages are panels, and everything else is JSON:

    GET  /                      the page
    GET  /api/state             everything the page needs to render
    POST /api/upload            replace the raw CSV (validated before replacing)
    POST /api/run/<stage>       start a background job (explore | clean | train)
    POST /api/cancel/<stage>    ask a running job to stop
    GET  /api/jobs              progress of every job, for polling
    POST /api/predict           score one pasted advert
    GET  /figure/<name>         a generated PNG

Routes stay thin: validate the request, call the library, return JSON. All
analysis lives in the sibling modules, so the CLI and this interface share one
implementation and cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from .. import analysis, charts, config, reset
from .. import preprocess as preprocess_module
from ..analysis import ANALYSIS_FIGURES, load_analysis
from ..data import (
    DatasetError,
    class_balance,
    clear_cache,
    load_raw,
    missingness,
    validate_csv,
)
from ..models import (
    ALL_METRICS,
    DEFAULT_SPLIT,
    EMBEDDING_DEFAULTS,
    IMBALANCE_STRATEGIES,
    POOLING_LABELS,
    REGISTRY,
    STRATEGY_LABELS,
    UI_FIGURES,
    available_models,
    embedding_settings,
    load_ui_model,
    read_corpus,
    saved_model_summary,
    split_settings,
    train_and_evaluate,
    validate_metric,
    validate_metrics,
    validate_model_names,
    validate_strategy,
)
from ..predict import corpus_options, predict_posting
from ..preprocess import DEFAULT_OPTIONS, read_corpus_meta, run_preprocess
from ..splits import (
    MAX_TEST_FRACTION,
    MIN_TEST_FRACTION,
    audit_split,
    build_split,
    leakage_report,
)
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

OPTION_HELP: dict[str, str] = {
    "normalise_placeholders": (
        "Collapse #EMAIL_<hash># to emailref. The hash identifies one "
        "specific scammer, so keeping it lets the model memorise rather than "
        "generalise — but the presence of a contact address is real signal "
        "(20.1% of fraudulent postings against 6.8% of legitimate)."
    ),
    "remove_stopwords": (
        "Drop the 318 most common English words, which carry no class signal."
    ),
    "stem": (
        "Conflate recruiting, recruiter and recruitment into one feature."
    ),
}

# Stages that run as background jobs. The report stages were
# command-line subcommands until now; the interface is the only
# surface, so they live here with the rest.
STAGES: tuple[str, ...] = (
    "clean",
    "train",
    *analysis.STAGE_KEYS,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def corpus_exists() -> bool:
    """Whether the cleaned corpus is on disk.

    Looked up through the module rather than bound at import time: a path
    imported by value is a snapshot, so anything that redirects the project's
    paths afterwards leaves this function answering about the old location.
    """
    return preprocess_module.CORPUS_PATH.exists()


def _to_rows(table: pd.DataFrame) -> dict[str, Any]:
    """A DataFrame as JSON-safe columns and rows.

    `object` cells coming from pandas can be numpy scalars, which `jsonify`
    refuses; everything is coerced through Python builtins here rather than at
    each call site.
    """
    def cell(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, (int, float)):
            return value
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (ValueError, AttributeError):
                pass
        if pd.isna(value):
            return None
        return str(value)

    return {
        "columns": [str(column) for column in table.columns],
        "rows": [[cell(value) for value in row] for row in table.to_numpy()],
    }


def dataset_summary() -> dict[str, Any] | None:
    """Row counts and class balance, or None when the CSV is absent or invalid."""
    try:
        frame = load_raw(copy=False)
    except DatasetError:
        return None
    balance = class_balance(frame)
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "balance": balance.to_dict("records"),
        "fraud_pct": float(balance.loc[1, "percent"]),
        "legit_pct": float(balance.loc[0, "percent"]),
        "n_fraud": int(balance.loc[1, "count"]),
    }


def save_results(payload: dict[str, Any]) -> None:
    config.ensure_directories()
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_results() -> dict[str, Any] | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("could not read %s", RESULTS_PATH.name)
        return None


def corpus_provenance() -> dict[str, Any] | None:
    """What built the stored corpus, formatted for display.

    Without this the interface cannot tell which options produced the stored
    corpus: a user could disable stemming, rebuild, restart, and then be shown
    the default settings while training on a corpus built differently. Any
    reported methodology would be wrong, and nothing would look broken.
    """
    meta = read_corpus_meta()
    if meta is None:
        return None
    options = {**DEFAULT_OPTIONS, **(meta.get("options") or {})}
    return {
        "built_at": meta.get("built_at", "unknown"),
        "rows": meta.get("rows"),
        "median_tokens": meta.get("median_tokens"),
        "vocabulary_stems": meta.get("vocabulary_stems"),
        "options": [
            {"key": key, "label": OPTION_LABELS.get(key, key), "enabled": bool(value)}
            for key, value in options.items()
            if key in OPTION_LABELS
        ],
    }


def _existing(names: tuple[str, ...]) -> list[dict[str, str]]:
    """Declared figures that have actually been generated, with captions."""
    output = []
    for name in names:
        path = config.FIGURES_DIR / name
        if path.exists():
            caption = (
                name.replace("ui_", "").replace(".png", "").replace("_", " ").strip()
            )
            output.append({
                "name": name,
                "caption": caption[:1].upper() + caption[1:],
                # Cache-buster: a regenerated figure keeps its filename, so
                # without this the browser shows the previous run's chart.
                "version": str(int(path.stat().st_mtime)),
            })
    return output


def _preview(frame: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    columns = ["title", "location", "has_company_logo", "telecommuting", "fraudulent"]
    present = [column for column in columns if column in frame.columns]
    preview = frame[present].head(n).copy()
    for column, width in (("title", 46), ("location", 26)):
        if column in preview.columns:
            preview[column] = preview[column].astype(str).str.slice(0, width)
    return preview


def _flag_table(frame: pd.DataFrame) -> pd.DataFrame:
    from ..explore import fraud_rate_by

    return pd.concat(
        [fraud_rate_by(frame, flag) for flag in config.BINARY_FLAGS], ignore_index=True
    )


def _corpus_sample(n: int = 3) -> list[dict[str, Any]]:
    if not corpus_exists():
        return []
    try:
        frame = read_corpus(copy=False)
    except (OSError, ValueError):
        return []
    if frame.empty:
        return []
    sample = frame.sample(min(n, len(frame)), random_state=config.RANDOM_STATE)
    return [
        {
            "label": "fraudulent" if row[config.TARGET] else "legitimate",
            "text": str(row["text_all"])[:320],
            "n_tokens": int(row["n_tokens"]),
        }
        for _, row in sample.iterrows()
    ]


def model_catalogue() -> list[dict[str, Any]]:
    """The models offered by the interface, with the metadata the page needs."""
    usable = available_models()
    return [
        {
            "name": spec.name,
            "label": spec.label,
            "blurb": spec.blurb,
            "family": spec.family,
            "supports_smote": spec.supports_smote,
            "interpretable": spec.has_coefficients,
            "cost": spec.cost,
            "available": name in usable,
            "needs_torch": spec.needs_torch,
        }
        for name, spec in REGISTRY.items()
    ]


def _split_summary(
    test_fraction: float = 0.2, grouped: bool = True, seed: int = config.RANDOM_STATE
) -> dict[str, Any] | None:
    """Compose and audit a split without training anything.

    This is what makes the evaluation protocol controllable rather than
    merely described: the user moves the slider, and the composition and the
    leakage measurement update before any model is fitted.
    """
    if not corpus_exists():
        return None
    try:
        # copy=False: everything below reads the frame, and a fresh copy per
        # call would defeat the group-key cache that makes this fast enough
        # to run on every slider release.
        frame = read_corpus(copy=False)
        train_idx, test_idx, provenance = build_split(
            frame, test_fraction=test_fraction, grouped=grouped, seed=seed
        )
        # strict=False: with grouping off the overlap is the point of the
        # exercise, so it is measured and shown rather than raised.
        table = audit_split(frame, train_idx, test_idx, strict=grouped)
        provenance.update(leakage_report(frame, train_idx, test_idx))
        return {
            "table": _to_rows(table),
            "provenance": provenance,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_total": int(len(frame)),
        }
    except Exception:
        logger.exception("could not audit the split")
        return None


_EXAMPLE_POSTING: dict[str, str] = {
    "title": "Data Entry Clerk - Work From Home",
    "description": (
        "Earn $5000 per month working from home! No experience necessary. "
        "We are recruiting motivated individuals to start immediately. "
        "Apply using the link below and send your details to our recruiter."
    ),
    "requirements": "Must have a computer and internet access. High school diploma.",
    "company_profile": "",
    "benefits": "",
}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    # A fixed development key would let anyone who read the source forge a
    # session cookie. There are no sessions here, but the key is still
    # generated per process rather than shipped as a constant.
    app.config["SECRET_KEY"] = os.environ.get("FJPD_SECRET_KEY") or secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["JSON_SORT_KEYS"] = False
    config.ensure_directories()

    # ------------------------------------------------------------- the page
    @app.get("/")
    def index():
        return render_template(
            "index.html",
            option_labels=OPTION_LABELS,
            option_help=OPTION_HELP,
            all_metrics=ALL_METRICS,
            strategies=IMBALANCE_STRATEGIES,
            strategy_labels=STRATEGY_LABELS,
            pooling_labels=POOLING_LABELS,
            embedding_defaults=EMBEDDING_DEFAULTS,
            models=model_catalogue(),
            example=_EXAMPLE_POSTING,
            text_fields=config.TEXT_FIELDS,
            binary_flags=config.BINARY_FLAGS,
            chart_catalogue=charts.catalogue(),
            analysis_catalogue=analysis.catalogue(),
        )

    # --------------------------------------------------------------- state
    @app.get("/api/state")
    def api_state():
        """The cheap facts the whole page depends on.

        Deliberately small. The previous version returned everything any
        panel might need -- the split audit, the corpus sample, the explore
        tables -- and the page rendered nothing until it arrived. The split
        audit alone was 4.3 of its 4.9 seconds, computed on every page load
        for a panel most visits never open.

        Everything here is a stat() call, a short JSON read, or an in-memory
        lookup. Panel data is fetched by the panel that needs it.
        """
        saved = saved_model_summary()
        name = saved["name"] if saved else None
        return jsonify({
            "dataset": dataset_summary(),
            "raw_path": str(config.RAW_CSV),
            "corpus_ready": corpus_exists(),
            "provenance": corpus_provenance(),
            "clean_options": [
                {
                    "key": key,
                    "label": OPTION_LABELS.get(key, key),
                    "help": OPTION_HELP.get(key, ""),
                    "default": bool(value),
                }
                for key, value in DEFAULT_OPTIONS.items()
            ],
            "split_defaults": {
                **DEFAULT_SPLIT,
                "min_test_fraction": MIN_TEST_FRACTION,
                "max_test_fraction": MAX_TEST_FRACTION,
            },
            "results": load_results(),
            "analysis": load_analysis(),
            "train_figures": _existing(UI_FIGURES),
            "analysis_figures": _existing(ANALYSIS_FIGURES),
            "model": {
                "name": name,
                "label": REGISTRY[name].label if name in REGISTRY else name,
                "metadata": saved["metadata"] if saved else None,
            },
            "cleaning_options_in_use": corpus_options(),
            "jobs": jobs.snapshots(),
            "active_job": jobs.active(),
        })

    @app.get("/api/explore/tables")
    def api_explore_tables():
        """The raw-data tables for the Explore panel, on demand."""
        try:
            frame = load_raw(copy=False)
        except DatasetError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        return jsonify({
            "ok": True,
            "preview": _to_rows(_preview(frame)),
            "tables": {
                "Missing values by column": _to_rows(missingness(frame).head(12)),
                "Fraud rate by binary flag": _to_rows(_flag_table(frame)),
            },
        })

    @app.get("/api/corpus/sample")
    def api_corpus_sample():
        """A few cleaned documents for the Clean panel, on demand."""
        if not corpus_exists():
            return jsonify({"ok": False, "error": "No corpus yet."}), 409
        return jsonify({"ok": True, "sample": _corpus_sample()})

    # -------------------------------------------------------------- upload
    @app.post("/api/upload")
    def api_upload():
        uploaded = request.files.get("csv")
        if uploaded is None or not uploaded.filename:
            return jsonify({"ok": False, "error": "No file selected."}), 400
        if not uploaded.filename.lower().endswith(".csv"):
            return jsonify({"ok": False, "error": "Expected a .csv file."}), 400

        # Validate before replacing. The previous version saved straight over
        # data/raw/fake_job_postings.csv and only then validated, so uploading
        # the wrong file destroyed a working dataset and left the app with
        # nothing to fall back on.
        config.ensure_directories()
        staging = config.RAW_CSV.with_suffix(".csv.incoming")
        try:
            uploaded.save(staging)
            frame = validate_csv(staging)
            # Only now replace the working file, and do it with a rename so
            # there is no window in which the dataset is half-written.
            staging.replace(config.RAW_CSV)
            clear_cache()
            charts.clear_cache()
            return jsonify({
                "ok": True,
                "message": (
                    f"Loaded and validated {uploaded.filename}: "
                    f"{len(frame):,} rows x {frame.shape[1]} columns."
                ),
            })
        except DatasetError as exc:
            staging.unlink(missing_ok=True)
            return jsonify({
                "ok": False,
                "error": f"Validation failed, your existing dataset was kept.\n\n{exc}",
            }), 400
        except OSError as exc:
            staging.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": f"Could not save the upload: {exc}"}), 500

    # ----------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    def api_jobs():
        return jsonify({"jobs": jobs.snapshots(), "active": jobs.active()})

    @app.post("/api/cancel/<stage>")
    def api_cancel(stage: str):
        if stage not in STAGES:
            return jsonify({"ok": False, "error": f"Unknown stage {stage!r}."}), 404
        if jobs.cancel(stage):
            return jsonify({"ok": True, "message": f"Cancelling {stage}…"})
        return jsonify({"ok": False, "error": f"{stage} is not running."}), 409

    @app.post("/api/run/<stage>")
    def api_run(stage: str):
        if stage not in STAGES:
            return jsonify({"ok": False, "error": f"Unknown stage {stage!r}."}), 404
        payload = request.get_json(silent=True) or {}
        if stage in analysis.STAGE_KEYS:
            builder = _analysis_task(stage)
        else:
            builder = {"clean": _clean_task, "train": _train_task}[stage]
        try:
            task = builder(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        started, message = jobs.start(stage, task)
        return jsonify({"ok": started, "message" if started else "error": message}), (
            200 if started else 409
        )

    # ---------------------------------------------------------------- reset
    @app.get("/api/reset/preview")
    def api_reset_preview():
        """What a reset would remove, so the dialog can show it before asking."""
        return jsonify({
            "ok": True,
            "targets": reset.describe(),
            "active_job": jobs.active(),
        })

    @app.post("/api/reset")
    def api_reset():
        """Delete generated artifacts and clear every cache.

        Refused while a job is running: a worker part-way through writing a
        corpus or a model would be left writing into a directory being
        emptied, and the resulting half-file would fail to load later with no
        hint as to why.
        """
        running = jobs.active()
        if running is not None:
            return jsonify({
                "ok": False,
                "error": f"'{running}' is still running. Wait for it to "
                         "finish or cancel it before resetting.",
            }), 409

        payload = request.get_json(silent=True) or {}
        include_dataset = bool(payload.get("include_dataset"))
        try:
            result = reset.perform(include_dataset=include_dataset)
        except OSError as exc:
            logger.exception("reset failed")
            return jsonify({"ok": False, "error": f"Reset failed: {exc}"}), 500

        jobs.reset_registry()
        kept = "" if include_dataset else " The imported dataset was kept."
        return jsonify({
            "ok": True,
            "message": (
                f"Removed {result['n_removed']} files "
                f"({result['bytes_freed'] / 1_048_576:.1f} MB).{kept}"
            ),
            **result,
        })

    # --------------------------------------------------------------- charts
    @app.post("/api/explore/charts")
    def api_explore_charts():
        """Data for the interactive Explore charts.

        Separate from /api/state because the user chooses which charts to
        show: computing all of them on every page render would make simply
        opening the app a multi-second wait for charts nobody asked for.
        """
        if dataset_summary() is None:
            return jsonify({"ok": False, "error": "Import the dataset first."}), 409
        payload = request.get_json(silent=True) or {}
        keys = charts.validate_keys(list(payload.get("charts") or []))
        if not keys:
            return jsonify({"ok": True, "charts": {}})
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            options = {}
        return jsonify({"ok": True, "charts": charts.build(keys, options)})

    # ---------------------------------------------------------------- split
    @app.post("/api/split/preview")
    def api_split_preview():
        """Compose a split and report what it looks like, without training.

        Cheap enough to call on every slider release: it factorises the
        description column and counts, which is a fraction of a second even
        at 17,880 rows.
        """
        if not corpus_exists():
            return jsonify({"ok": False, "error": "Build the corpus first."}), 409
        payload = request.get_json(silent=True) or {}
        settings = split_settings(
            test_fraction=payload.get("test_fraction"),
            grouped=payload.get("grouped"),
            seed=payload.get("seed"),
        )
        summary = _split_summary(
            test_fraction=settings["test_fraction"],
            grouped=settings["grouped"],
            seed=settings["seed"],
        )
        if summary is None:
            return jsonify({"ok": False, "error": "Could not build that split."}), 400
        return jsonify({"ok": True, **summary})

    # -------------------------------------------------------------- predict
    @app.post("/api/predict")
    def api_predict():
        saved = load_ui_model()
        if saved is None:
            return jsonify({
                "ok": False,
                "error": "No saved model. Train one first — prediction uses the "
                         "best model from your last training run.",
            }), 409

        name, pipeline, _metadata = saved
        payload = request.get_json(silent=True) or {}
        values: dict[str, Any] = {
            field: str(payload.get(field, "") or "") for field in config.TEXT_FIELDS
        }
        for flag in config.BINARY_FLAGS:
            values[flag] = 1 if payload.get(flag) else 0

        if not values["description"].strip():
            return jsonify({"ok": False, "error": "A description is required."}), 400

        try:
            result = predict_posting(
                pipeline,
                values,
                include_sensitivity=bool(payload.get("sensitivity", True)),
            )
        except Exception as exc:
            logger.exception("prediction failed")
            return jsonify({
                "ok": False,
                "error": f"Prediction failed: {type(exc).__name__}: {exc}",
            }), 500

        label = REGISTRY[name].label if name in REGISTRY else name
        return jsonify({"ok": True, "model": name, "model_label": label, "result": result})

    # -------------------------------------------------------------- figures
    @app.get("/figure/<path:filename>")
    def figure(filename: str):
        # send_from_directory rejects paths that escape the directory, so a
        # traversal attempt in `filename` cannot reach outside reports/figures.
        if Path(filename).suffix.lower() != ".png":
            return jsonify({"error": "Only PNG figures are served."}), 404
        return send_from_directory(config.FIGURES_DIR, filename)

    @app.errorhandler(413)
    def too_large(_error):
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        return jsonify({"ok": False, "error": f"File exceeds the {limit} MB limit."}), 413

    @app.errorhandler(500)
    def server_error(error):  # pragma: no cover - defensive
        logger.exception("unhandled error", exc_info=error)
        return jsonify({"ok": False, "error": "Internal error; see the server log."}), 500

    return app


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------
def _analysis_task(stage: str):
    """Build a runner for one of the report stages."""
    runner = analysis.RUNNERS[stage]

    def builder(payload: dict[str, Any]):
        if stage != "report_eda" and not corpus_exists():
            raise ValueError("Build the cleaned corpus first.")
        params = dict(payload or {})

        def task(reporter):
            return runner(params, reporter)

        return task

    return builder


def _clean_task(payload: dict[str, Any]):
    requested = payload.get("options") or {}
    options = {key: bool(requested.get(key, default))
               for key, default in DEFAULT_OPTIONS.items()}

    def task(reporter):
        run_preprocess(progress=reporter, **options)
        return {"options": options}

    return task


def _train_task(payload: dict[str, Any]):
    selected = validate_model_names(list(payload.get("models") or []))
    if not selected:
        raise ValueError("Select at least one model.")
    if not corpus_exists():
        raise ValueError("Build the cleaned corpus first.")

    metrics = validate_metrics(list(payload.get("metrics") or []))
    strategy = validate_strategy(str(payload.get("strategy", "class_weight")))
    rank_by = validate_metric(str(payload.get("rank_by", "f1_fraud")))
    embedding = embedding_settings(**(payload.get("embedding") or {}))
    split = split_settings(**(payload.get("split") or {}))

    def task(reporter):
        loading = reporter.span(0.04, "loading the cleaned corpus")
        loading(0.2, "loading the cleaned corpus")
        frame = read_corpus()
        loading.done(f"loaded {len(frame):,} postings")

        result = train_and_evaluate(
            frame,
            selected,
            strategy=strategy,
            rank_by=rank_by,
            embedding_params=embedding,
            split_params=split,
            progress=reporter.span(1.0, "training"),
        )
        result["metrics"] = metrics
        save_results(result)
        return result

    return task
