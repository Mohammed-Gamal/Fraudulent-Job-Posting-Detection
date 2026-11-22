"""Stages 3-4: feature extraction and model comparison.

Everything is expressed as a scikit-learn Pipeline so that each transform is
fitted inside the cross-validation fold. Vectorising the whole corpus before
splitting would let test-set vocabulary and document frequencies inform
training -- a subtler leak than duplicate rows, and just as damaging.

Two feature families
--------------------
`sparse`    TF-IDF over the cleaned text, optionally concatenated with the
            structured flags. All blocks are non-negative, which keeps
            Multinomial Naive Bayes applicable alongside the discriminative
            models.

`embedding` Integer token sequences feeding a trainable embedding layer
            (see `embeddings.py`). Dense, signed, and trained by gradient
            descent rather than a closed-form fit.

Both families are wrapped in the same `ColumnTransformer -> estimator`
pipeline, so one cross-validation routine with one set of folds evaluates
both. That is the only way the comparison between them means anything.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from . import config
from .progress import ProgressReporter
from .splits import cv_splitter, make_group_key

logger = logging.getLogger(__name__)

TEXT_COLUMN = "text_all"

# Structured features: the binary flags plus the missingness indicators that
# Stage 1 showed to be strongly associated with fraud.
STRUCTURED_BINARY: list[str] = config.BINARY_FLAGS + [
    f"{field_name}_is_missing"
    for field_name in config.TEXT_FIELDS + config.CATEGORICAL_FIELDS
]
STRUCTURED_NUMERIC: list[str] = ["n_tokens"]

SCORING = {
    "f1_fraud": "f1",
    "precision_fraud": "precision",
    "recall_fraud": "recall",
    "avg_precision": "average_precision",
    "roc_auc": "roc_auc",
}

ALL_METRICS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision_fraud": "Precision (fraud)",
    "recall_fraud": "Recall (fraud)",
    "f1_fraud": "F1 (fraud)",
    "avg_precision": "Average precision",
    "roc_auc": "ROC AUC",
}

IMBALANCE_STRATEGIES: tuple[str, ...] = ("none", "class_weight", "smote")

STRATEGY_LABELS: dict[str, str] = {
    "none": "No treatment",
    "class_weight": "Class weighting",
    "smote": "SMOTE oversampling",
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Everything the rest of the project needs to know about one model.

    Keeping this in one table is what lets the interface validate a submitted
    model name, decide which controls to disable, and label results correctly
    without a chain of special cases spread across the routes.
    """

    name: str
    family: str                  # "sparse" | "embedding"
    label: str
    blurb: str
    supports_smote: bool = True
    has_coefficients: bool = False
    # Relative cost, used to weight the progress bar across selected models.
    cost: float = 1.0
    needs_torch: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)


REGISTRY: dict[str, ModelSpec] = {
    "MultinomialNB": ModelSpec(
        name="MultinomialNB",
        family="sparse",
        label="Multinomial Naive Bayes",
        blurb=(
            "Generative baseline over term counts. Has no class-weight "
            "parameter, so it cannot be rebalanced the way the others can -- "
            "which is itself a finding worth reporting."
        ),
        has_coefficients=True,
        cost=0.4,
    ),
    "LogisticRegression": ModelSpec(
        name="LogisticRegression",
        family="sparse",
        label="Logistic Regression",
        blurb="Linear, probabilistic, and directly interpretable by coefficient.",
        has_coefficients=True,
        cost=1.2,
    ),
    "LinearSVC": ModelSpec(
        name="LinearSVC",
        family="sparse",
        label="Linear SVC",
        blurb=(
            "Maximum-margin linear classifier. Strongest sparse model on this "
            "dataset and the reference point for the embedding model."
        ),
        has_coefficients=True,
        cost=1.0,
    ),
    "RandomForest": ModelSpec(
        name="RandomForest",
        family="sparse",
        label="Random Forest",
        blurb=(
            "Non-linear ensemble. Can capture flag interactions a linear model "
            "cannot, at the cost of per-term interpretability."
        ),
        has_coefficients=False,
        cost=3.0,
    ),
    "EmbeddingNet": ModelSpec(
        name="EmbeddingNet",
        family="embedding",
        label="Embedding network",
        blurb=(
            "Trainable embedding layer, attention pooling, then a dense head. "
            "Learns a dense vector per term instead of treating every term as "
            "an unrelated dimension, so it can generalise across synonyms."
        ),
        # Interpolating between integer token ids produces ids that mean
        # nothing, so SMOTE is incoherent here. The network rebalances through
        # pos_weight in its loss instead.
        supports_smote=False,
        has_coefficients=False,
        cost=4.0,
        needs_torch=True,
    ),
}


def available_models(include_unavailable: bool = False) -> dict[str, ModelSpec]:
    """Models usable in this environment.

    The embedding model is hidden rather than offered-and-broken when torch is
    absent, so the interface cannot present a control that cannot work.
    """
    from .embeddings import torch_available

    torch_ok = torch_available()
    return {
        name: spec
        for name, spec in REGISTRY.items()
        if include_unavailable or not spec.needs_torch or torch_ok
    }


def validate_model_names(names: list[str]) -> list[str]:
    """Keep submitted model names that exist and are usable, in registry order.

    Form input is untrusted: a stale page, an edited request or a renamed model
    would otherwise reach `REGISTRY[name]` and raise a 500 well away from the
    cause.
    """
    usable = available_models()
    accepted = [name for name in usable if name in set(names)]
    rejected = sorted(set(names) - set(usable))
    if rejected:
        logger.warning("ignoring unknown or unavailable models: %s", rejected)
    return accepted


def validate_strategy(strategy: str) -> str:
    return strategy if strategy in IMBALANCE_STRATEGIES else "class_weight"


def validate_metric(metric: str, default: str = "f1_fraud") -> str:
    return metric if metric in ALL_METRICS else default


def validate_metrics(metrics: list[str]) -> list[str]:
    """Metric columns to display, in canonical order, never empty."""
    chosen = [key for key in ALL_METRICS if key in set(metrics)]
    return chosen or ["f1_fraud"]


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------
def _structured_blocks() -> list[tuple[str, Any, Any]]:
    return [
        ("binary", "passthrough", STRUCTURED_BINARY),
        ("numeric", MinMaxScaler(), STRUCTURED_NUMERIC),
    ]


def build_features(
    max_features: int = 30_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    use_structured: bool = True,
) -> ColumnTransformer:
    """TF-IDF over the cleaned text, optionally concatenated with structured features.

    sublinear_tf dampens repeated boilerplate, which matters here because many
    postings repeat phrases heavily within a single document.
    """
    blocks: list[tuple[str, Any, Any]] = [
        (
            "text",
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                sublinear_tf=True,
            ),
            TEXT_COLUMN,
        )
    ]
    if use_structured:
        blocks.extend(_structured_blocks())
    return ColumnTransformer(blocks, sparse_threshold=1.0)


# Defaults chosen by a sweep on the held-out split, recorded in the README.
# Notably sequence_length=150 beat 200 and 300: truncating to the opening of
# the advert concentrates the signal instead of diluting it across padding and
# boilerplate, and it roughly halves training time.
EMBEDDING_DEFAULTS: dict[str, Any] = {
    "sequence_length": 150,
    "vocabulary_size": 20_000,
    "embedding_dim": 64,
    "hidden_dim": 64,
    "n_filters": 64,
    "dropout": 0.4,
    "pooling": "conv",
    "learning_rate": 1.5e-3,
    # Kept very low on purpose: Adam's weight decay shrinks every embedding
    # row, including the rare terms that carry the most signal, and raising it
    # to 1e-3 cost 8 points of F1 in testing.
    "weight_decay": 1e-5,
    "batch_size": 128,
    "max_epochs": 15,
    "patience": 4,
    "min_document_frequency": 3,
    "tune_threshold": True,
}

POOLING_LABELS: dict[str, str] = {
    "conv": "Convolutional n-grams (1-4)",
    "gated": "Gated mean (learned term weights)",
    "attention": "Additive attention",
    "mean": "Masked mean",
    "max": "Masked max",
}


def embedding_settings(**overrides: Any) -> dict[str, Any]:
    """Embedding hyperparameters with user overrides applied and bounded.

    The interface exposes these, so every value is clamped to a range that
    cannot exhaust memory or run for an unbounded time on a laptop.
    """
    settings = {**EMBEDDING_DEFAULTS}
    for key, value in overrides.items():
        if key in settings and value is not None:
            settings[key] = value

    def clamp(key: str, low: Any, high: Any, cast) -> None:
        try:
            settings[key] = cast(min(max(cast(settings[key]), low), high))
        except (TypeError, ValueError):
            settings[key] = EMBEDDING_DEFAULTS[key]

    clamp("sequence_length", 32, 1000, int)
    clamp("vocabulary_size", 1_000, 60_000, int)
    clamp("embedding_dim", 8, 256, int)
    clamp("hidden_dim", 8, 256, int)
    clamp("n_filters", 8, 256, int)
    clamp("dropout", 0.0, 0.8, float)
    clamp("learning_rate", 1e-5, 0.1, float)
    clamp("weight_decay", 0.0, 0.1, float)
    clamp("batch_size", 8, 1024, int)
    clamp("max_epochs", 1, 60, int)
    clamp("patience", 1, 20, int)
    clamp("min_document_frequency", 1, 50, int)
    settings["tune_threshold"] = bool(settings["tune_threshold"])
    from .embeddings import POOLING_MODES

    if settings["pooling"] not in POOLING_MODES:
        settings["pooling"] = EMBEDDING_DEFAULTS["pooling"]
    return settings


def build_embedding_features(
    sequence_length: int,
    vocabulary_size: int,
    min_document_frequency: int,
    use_structured: bool = True,
) -> ColumnTransformer:
    """Token-id sequences, optionally concatenated with structured features.

    `sparse_threshold=0.0` forces a dense result: the embedding layer indexes
    into a weight matrix and cannot consume a sparse matrix.
    """
    from .embeddings import SequenceEncoder

    blocks: list[tuple[str, Any, Any]] = [
        (
            "sequence",
            SequenceEncoder(
                max_vocabulary=vocabulary_size,
                sequence_length=sequence_length,
                min_document_frequency=min_document_frequency,
            ),
            TEXT_COLUMN,
        )
    ]
    if use_structured:
        blocks.extend(_structured_blocks())
    return ColumnTransformer(blocks, sparse_threshold=0.0)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def linear_svc(balanced: bool) -> LinearSVC:
    """The tuned LinearSVC used as the sparse reference model throughout.

    max_iter is far above the default because liblinear does not converge on
    this feature matrix at 1,000 iterations.
    """
    return LinearSVC(
        class_weight="balanced" if balanced else None,
        random_state=config.RANDOM_STATE,
        dual="auto",
        max_iter=20_000,
    )


def build_estimator(
    name: str,
    balanced: bool = True,
    embedding_params: dict[str, Any] | None = None,
    progress: ProgressReporter | None = None,
) -> Any:
    """One bare estimator by registry name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    weight = "balanced" if balanced else None

    if name == "MultinomialNB":
        return MultinomialNB(alpha=0.1)
    if name == "LogisticRegression":
        return LogisticRegression(
            max_iter=2000, class_weight=weight, random_state=config.RANDOM_STATE
        )
    if name == "LinearSVC":
        return linear_svc(balanced)
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight=weight,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
    if name == "EmbeddingNet":
        from .embeddings import EmbeddingTextClassifier

        settings = embedding_settings(**(embedding_params or {}))
        return EmbeddingTextClassifier(
            sequence_length=settings["sequence_length"],
            vocabulary_size=settings["vocabulary_size"],
            embedding_dim=settings["embedding_dim"],
            hidden_dim=settings["hidden_dim"],
            n_filters=settings["n_filters"],
            dropout=settings["dropout"],
            pooling=settings["pooling"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            batch_size=settings["batch_size"],
            max_epochs=settings["max_epochs"],
            patience=settings["patience"],
            tune_threshold=settings["tune_threshold"],
            balanced=balanced,
            random_state=config.RANDOM_STATE,
            progress=progress,
        )
    raise KeyError(name)  # pragma: no cover - guarded above


def candidate_models(balanced: bool = True) -> dict[str, Any]:
    """Every usable model as a bare estimator, keyed by name."""
    return {
        name: build_estimator(name, balanced=balanced)
        for name in available_models()
    }


def build_pipeline(
    name: str,
    strategy: str = "class_weight",
    use_structured: bool = True,
    embedding_params: dict[str, Any] | None = None,
    progress: ProgressReporter | None = None,
) -> tuple[Any, str]:
    """A fittable pipeline for `name`, plus the strategy actually applied.

    The returned strategy can differ from the one requested: SMOTE is
    incoherent for integer token sequences, so the embedding model falls back
    to loss weighting. Returning what was really used stops the interface from
    labelling results with a treatment that never ran.
    """
    spec = REGISTRY[name]
    strategy = validate_strategy(strategy)

    if strategy == "smote" and not spec.supports_smote:
        logger.info(
            "%s does not support SMOTE (token ids cannot be interpolated); "
            "using loss weighting instead", name,
        )
        strategy = "class_weight"

    balanced = strategy == "class_weight"
    estimator = build_estimator(
        name, balanced=balanced, embedding_params=embedding_params, progress=progress
    )

    if spec.family == "embedding":
        settings = embedding_settings(**(embedding_params or {}))
        features = build_embedding_features(
            sequence_length=settings["sequence_length"],
            vocabulary_size=settings["vocabulary_size"],
            min_document_frequency=settings["min_document_frequency"],
            use_structured=use_structured,
        )
    else:
        features = build_features(use_structured=use_structured)

    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        pipeline = ImbPipeline([
            ("features", features),
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            ("clf", estimator),
        ])
    else:
        pipeline = Pipeline([("features", features), ("clf", estimator)])
    return pipeline, strategy


def make_pipeline(estimator: Any, **feature_kwargs: Any) -> Pipeline:
    """Sparse-family pipeline around an already-built estimator."""
    return Pipeline([("features", build_features(**feature_kwargs)), ("clf", estimator)])


def wrap_strategy(estimator: Any, strategy: str) -> Any:
    """Attach an imbalance treatment around a bare sparse estimator."""
    features = build_features()
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline([
            ("features", features),
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            ("clf", estimator),
        ])
    return Pipeline([("features", features), ("clf", estimator)])


def score_kind(pipeline: Any) -> str:
    """Whether this pipeline's scores are signed margins or probabilities.

    The decision boundary differs -- 0.0 for a margin, 0.5 for a probability --
    so a threshold plot or a reported operating point that assumes one for the
    other is simply mislabelled.
    """
    return "margin" if hasattr(pipeline, "decision_function") else "probability"


def default_threshold(kind: str) -> float:
    """Where `predict` cuts, for an estimator that has not tuned a threshold."""
    return 0.0 if kind == "margin" else 0.5


def operating_threshold(pipeline: Any) -> float:
    """The threshold this fitted pipeline's `predict` actually applies.

    The embedding network tunes its own cut point on its early-stopping
    holdout, so assuming the family default would misreport its operating
    point everywhere it is displayed.
    """
    classifier = getattr(pipeline, "named_steps", {}).get("clf")
    tuned = getattr(classifier, "threshold_", None)
    if tuned is not None:
        return float(tuned)
    return default_threshold(score_kind(pipeline))


def continuous_scores(pipeline: Any, frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Continuous scores plus their kind, whichever interface the estimator has."""
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(frame)).ravel(), "margin"
    return np.asarray(pipeline.predict_proba(frame))[:, 1], "probability"


# ---------------------------------------------------------------------------
# Cross-validation (CLI, report numbers)
# ---------------------------------------------------------------------------
def cross_validate_models(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    names: list[str] | None = None,
    use_structured: bool = True,
    n_splits: int = config.CV_FOLDS,
) -> pd.DataFrame:
    """Grouped, stratified cross-validation of each candidate on the training set."""
    names = list(names) if names is not None else list(available_models())
    train = frame.iloc[train_idx]
    labels = train[config.TARGET].to_numpy()
    groups = make_group_key(train).to_numpy()

    rows: list[dict[str, Any]] = []
    for name in names:
        pipeline, _ = build_pipeline(
            name, strategy="class_weight", use_structured=use_structured
        )
        # LinearSVC exposes decision_function but not predict_proba; the
        # ranking metrics accept either, so no special casing is needed.
        scores = cross_validate(
            pipeline,
            train,
            labels,
            groups=groups,
            cv=cv_splitter(n_splits),
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        row: dict[str, Any] = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[metric] = round(float(values.mean()), 4)
            row[f"{metric}_sd"] = round(float(values.std()), 4)
        row["fit_seconds"] = round(float(scores["fit_time"].mean()), 1)
        rows.append(row)
        logger.info(
            "%-20s F1=%.4f  AP=%.4f  (%.0fs/fit)",
            name, row["f1_fraud"], row["avg_precision"], row["fit_seconds"],
        )
    return pd.DataFrame(rows)


RESULTS_PATH = config.INTERIM_DIR / "cv_results.csv"

FEATURE_SETS: dict[str, bool] = {
    "tfidf + structured": True,
    "tfidf only": False,
}


def _completed() -> set[tuple[str, str]]:
    if not RESULTS_PATH.exists():
        return set()
    try:
        done = pd.read_csv(RESULTS_PATH)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        logger.warning("could not read %s; starting over", RESULTS_PATH.name)
        return set()
    if not {"model", "features"} <= set(done.columns):
        return set()
    return set(zip(done["model"], done["features"], strict=True))


def run_comparison(only: str | None = None) -> pd.DataFrame:
    """Stage 4a: cross-validate every candidate, resuming where it left off.

    Each completed (model, feature set) pair is appended to disk immediately,
    so an interrupted run loses at most the model in flight.
    """
    from .plotting import save_table
    from .splits import train_test_indices

    config.ensure_directories()
    frame = read_corpus(copy=False)
    train_idx, _ = train_test_indices(frame)

    done = _completed()
    for feature_label, use_structured in FEATURE_SETS.items():
        for name in available_models():
            if only is not None and name != only:
                continue
            if (name, feature_label) in done:
                logger.info("skip %s / %s (already done)", name, feature_label)
                continue
            result = cross_validate_models(
                frame, train_idx, [name], use_structured=use_structured
            )
            result["features"] = feature_label
            result.to_csv(
                RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False
            )

    table = pd.read_csv(RESULTS_PATH)
    ordered = ["model", "features", "f1_fraud", "f1_fraud_sd", "precision_fraud",
               "recall_fraud", "avg_precision", "roc_auc", "fit_seconds"]
    table = table[ordered].sort_values("f1_fraud", ascending=False).reset_index(drop=True)
    save_table(table, "model_comparison_cv")
    return table


def run_imbalance_comparison(
    n_splits: int = config.CV_FOLDS, progress: Any = None
) -> pd.DataFrame:
    """Stage 4b: does the imbalance remedy matter, and which one?

    Compares no treatment, inverse-frequency class weighting and SMOTE on the
    strongest sparse classifier. SMOTE is applied inside the pipeline, so it is
    fitted on the training folds only -- oversampling before cross-validation
    would place synthetic copies of validation examples into training.
    """
    from .plotting import save_table
    from .splits import train_test_indices

    reporter = progress if progress is not None else ProgressReporter()
    config.ensure_directories()
    frame = read_corpus(copy=False)
    train_idx, _ = train_test_indices(frame)
    train = frame.iloc[train_idx]
    labels = train[config.TARGET].to_numpy()
    groups = make_group_key(train).to_numpy()

    strategies = {
        "none": wrap_strategy(linear_svc(False), "none"),
        "class_weight=balanced": wrap_strategy(linear_svc(True), "class_weight"),
        "SMOTE": wrap_strategy(linear_svc(False), "smote"),
    }

    rows: list[dict[str, Any]] = []
    spans = reporter.span(0.95, "strategies").weighted(
        dict.fromkeys(strategies, 1.0)
    )
    for name, pipeline in strategies.items():
        spans[name](0.05, f"cross-validating {name}")
        scores = cross_validate(
            pipeline, train, labels, groups=groups, cv=cv_splitter(n_splits),
            scoring=SCORING, n_jobs=1, error_score="raise",
        )
        row: dict[str, Any] = {"strategy": name, "model": "LinearSVC"}
        row.update({m: round(float(scores[f"test_{m}"].mean()), 4) for m in SCORING})
        rows.append(row)
        spans[name].done(f"{name}: F1 {row['f1_fraud']:.3f}")
        logger.info("%-22s F1=%.4f  P=%.4f  R=%.4f",
                    name, row["f1_fraud"], row["precision_fraud"], row["recall_fraud"])

    table = pd.DataFrame(rows)
    save_table(table, "imbalance_comparison")
    return table


# ---------------------------------------------------------------------------
# Single-split training, used by the web interface
# ---------------------------------------------------------------------------
UI_MODEL_PATH = config.MODELS_DIR / "ui_model.joblib"
UI_MODEL_META_PATH = config.MODELS_DIR / "ui_model.json"

# Figures regenerated after every interactive training run, named apart from
# the CLI's so an exploratory run never overwrites report output.
UI_FIGURES: tuple[str, ...] = (
    "ui_confusion_matrix.png",
    "ui_precision_recall.png",
    "ui_threshold_tradeoff.png",
    "ui_embedding_training.png",
)


# The corpus is read by almost every request the interface serves. Parsing it
# each time cost a sixth of a second per call and several calls per page, so
# it is held in memory and keyed on the file's identity -- rebuilding the
# corpus changes its mtime and invalidates this without an explicit call.
_CORPUS_LOCK = threading.Lock()
_CORPUS_CACHE: dict[str, Any] = {"key": None, "frame": None}


def _corpus_key() -> tuple[int, int] | None:
    from .preprocess import CORPUS_PATH

    try:
        stat = CORPUS_PATH.stat()
    except OSError:
        return None
    return (stat.st_size, int(stat.st_mtime_ns))


def clear_corpus_cache() -> None:
    with _CORPUS_LOCK:
        _CORPUS_CACHE["key"] = None
        _CORPUS_CACHE["frame"] = None


def read_corpus(copy: bool = True) -> pd.DataFrame:
    """The cleaned corpus from disk, with a clear error if it is not built yet.

    `copy=False` hands back the cached frame itself. Only for callers that
    treat it as read-only; anything that mutates must take a copy or it will
    corrupt what every later request sees.
    """
    from .preprocess import CORPUS_PATH

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "No cleaned corpus found. Build it first on the Clean step."
        )

    key = _corpus_key()
    with _CORPUS_LOCK:
        if key is not None and _CORPUS_CACHE["key"] == key:
            frame = _CORPUS_CACHE["frame"]
            return frame.copy() if copy else frame

    frame = pd.read_parquet(CORPUS_PATH)
    with _CORPUS_LOCK:
        _CORPUS_CACHE["key"] = key
        _CORPUS_CACHE["frame"] = frame
    return frame.copy() if copy else frame


def _save_best(name: str, pipeline: Any, metadata: dict[str, Any]) -> None:
    """Persist the winning pipeline so the prediction panel has a model.

    Training results that vanish when the process restarts are of no use.

    Written to a temporary file and renamed into place. A dump that fails
    halfway -- which is exactly what an unpicklable attribute causes -- would
    otherwise overwrite a working model with a truncated file, and the failure
    would surface later as an EOFError on load rather than here.
    """
    import joblib

    config.ensure_directories()
    temporary = UI_MODEL_PATH.with_suffix(".joblib.tmp")
    try:
        joblib.dump(
            {"name": name, "pipeline": pipeline, "metadata": metadata}, temporary
        )
        temporary.replace(UI_MODEL_PATH)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    # A sidecar describing the model, so the interface can say which model is
    # loaded without unpickling the pipeline. Reading the .joblib just to show
    # a name cost ~130ms on every page render, and for the embedding model it
    # means rebuilding the network.
    import json

    UI_MODEL_META_PATH.write_text(
        json.dumps({"name": name, "metadata": metadata}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("best model %s saved to %s", name, UI_MODEL_PATH.name)


def saved_model_summary() -> dict[str, Any] | None:
    """Name and metadata of the saved model, without loading it.

    Falls back to unpickling only when the sidecar is missing, which happens
    for a model saved by an older version of this project.
    """
    import json

    if not UI_MODEL_PATH.exists():
        return None
    if UI_MODEL_META_PATH.exists():
        try:
            payload = json.loads(UI_MODEL_META_PATH.read_text(encoding="utf-8"))
            return {
                "name": payload.get("name"),
                "metadata": payload.get("metadata") or {},
            }
        except (OSError, json.JSONDecodeError):
            logger.warning("could not read %s", UI_MODEL_META_PATH.name)

    # No sidecar: a model saved before this file existed. Unpickle once to
    # recover the name, then write the sidecar so the next request -- and
    # every page render after it -- is a small JSON read instead.
    saved = load_ui_model()
    if saved is None:
        return None
    summary = {"name": saved[0], "metadata": saved[2]}
    try:
        UI_MODEL_META_PATH.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
    except OSError:
        logger.debug("could not backfill %s", UI_MODEL_META_PATH.name)
    return summary


def load_ui_model() -> tuple[str, Any, dict[str, Any]] | None:
    """The pipeline saved by the last interactive training run, if any."""
    if not UI_MODEL_PATH.exists():
        return None
    import joblib

    try:
        payload = joblib.load(UI_MODEL_PATH)
        return payload["name"], payload["pipeline"], payload.get("metadata", {})
    except Exception:
        logger.exception("could not load %s", UI_MODEL_PATH)
        return None


DEFAULT_SPLIT: dict[str, Any] = {
    "test_fraction": 0.2,
    "grouped": True,
    "seed": config.RANDOM_STATE,
}


def split_settings(**overrides: Any) -> dict[str, Any]:
    """Split configuration with user overrides applied and bounded."""
    from .splits import MAX_TEST_FRACTION, MIN_TEST_FRACTION

    settings = {**DEFAULT_SPLIT}
    for key, value in overrides.items():
        if key in settings and value is not None:
            settings[key] = value
    try:
        fraction = float(settings["test_fraction"])
    except (TypeError, ValueError):
        fraction = DEFAULT_SPLIT["test_fraction"]
    settings["test_fraction"] = min(
        max(fraction, MIN_TEST_FRACTION), MAX_TEST_FRACTION
    )
    settings["grouped"] = bool(settings["grouped"])
    try:
        settings["seed"] = int(settings["seed"])
    except (TypeError, ValueError):
        settings["seed"] = config.RANDOM_STATE
    return settings


def train_and_evaluate(
    frame: pd.DataFrame,
    model_names: list[str],
    strategy: str = "class_weight",
    rank_by: str = "f1_fraud",
    embedding_params: dict[str, Any] | None = None,
    split_params: dict[str, Any] | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Fit each selected model on the training split and score it on the test split.

    A single split rather than cross-validation: the interface needs to stay
    responsive, and one grouped split is enough to rank models. Use
    `fjpd compare` for the cross-validated numbers that belong in a report.

    Returns a payload the interface can render directly, including which
    imbalance strategy each model actually received and what kind of score it
    produces -- both of which differ between the sparse and embedding families.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from .splits import build_split, leakage_report

    reporter = progress if progress is not None else ProgressReporter()
    model_names = validate_model_names(model_names)
    if not model_names:
        raise ValueError("no usable models were selected")
    strategy = validate_strategy(strategy)
    rank_by = validate_metric(rank_by)
    split = split_settings(**(split_params or {}))

    reporter(0.02, "preparing the train/test split")
    train_idx, test_idx, split_provenance = build_split(
        frame,
        test_fraction=split["test_fraction"],
        grouped=split["grouped"],
        seed=split["seed"],
    )
    split_provenance.update(leakage_report(frame, train_idx, test_idx))
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    y_train = train[config.TARGET].to_numpy()
    y_test = test[config.TARGET].to_numpy()
    train_groups = make_group_key(train).to_numpy()

    def rounded(metric, *args, **kwargs) -> float:
        return round(float(metric(*args, **kwargs)), 4)

    # Reserve the last tenth of the bar for saving and plotting, and split the
    # rest across the models in proportion to how long each tends to take.
    fitting = reporter.span(0.88, "training")
    weights = {name: REGISTRY[name].cost for name in model_names}
    per_model = fitting.weighted(weights)

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    outputs: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    applied: dict[str, str] = {}
    notes: list[str] = []

    for name in model_names:
        spec = REGISTRY[name]
        step = per_model[name]
        step(0.0, f"fitting {spec.label}")

        # The embedding model reports its own epochs into the inner 80% of its
        # span; the closed-form models have nothing to report mid-fit.
        inner = step.span(0.8, spec.label) if spec.family == "embedding" else None
        pipeline, used = build_pipeline(
            name,
            strategy=strategy,
            embedding_params=embedding_params,
            progress=inner,
        )
        applied[name] = used
        if used != strategy:
            notes.append(
                f"{spec.label}: {STRATEGY_LABELS[strategy]} is not applicable, "
                f"used {STRATEGY_LABELS[used]} instead."
            )

        if spec.family == "embedding":
            # Group-disjoint early-stopping holdout: without the group array a
            # duplicated description could sit on both sides and the stopping
            # epoch would be chosen on a flattered score.
            pipeline.fit(train, y_train, clf__groups=train_groups)
        else:
            pipeline.fit(train, y_train)

        step(0.9, f"scoring {spec.label}")
        scores, kind = continuous_scores(pipeline, test)
        predictions = np.asarray(pipeline.predict(test)).ravel()

        fitted[name] = pipeline
        outputs[name] = (scores, predictions, kind)
        rows.append({
            "model": name,
            "label": spec.label,
            "family": spec.family,
            "strategy": used,
            "score_kind": kind,
            "threshold": round(operating_threshold(pipeline), 4),
            "accuracy": rounded(accuracy_score, y_test, predictions),
            "precision_fraud": rounded(precision_score, y_test, predictions, zero_division=0),
            "recall_fraud": rounded(recall_score, y_test, predictions, zero_division=0),
            "f1_fraud": rounded(f1_score, y_test, predictions, zero_division=0),
            "avg_precision": rounded(average_precision_score, y_test, scores),
            "roc_auc": rounded(roc_auc_score, y_test, scores),
        })
        step.done(f"{spec.label} done (F1 {rows[-1]['f1_fraud']:.3f})")
        logger.info("%s trained: F1=%.4f", name, rows[-1]["f1_fraud"])

    table = pd.DataFrame(rows).sort_values(rank_by, ascending=False)
    table = table.reset_index(drop=True)

    best = str(table.loc[0, "model"])
    best_scores, best_predictions, best_kind = outputs[best]

    plotting = reporter.span(1.0, f"saving {REGISTRY[best].label}")
    plotting(0.1, f"saving {REGISTRY[best].label}")

    # The convergence curve belongs to the embedding model whether or not it
    # won. Reading it off the winner lost it whenever a sparse model ranked
    # higher, which is most of the time.
    history: list[dict[str, float]] | None = None
    embedding_name = next(
        (name for name in model_names if REGISTRY[name].family == "embedding"), None
    )
    if embedding_name is not None:
        history = getattr(fitted[embedding_name].named_steps["clf"], "history_", None)

    _save_best(
        best,
        fitted[best],
        {
            "strategy": applied[best],
            "score_kind": best_kind,
            "family": REGISTRY[best].family,
            "embedding_params": (
                embedding_settings(**(embedding_params or {}))
                if REGISTRY[best].family == "embedding"
                else None
            ),
        },
    )

    plotting(0.4, "drawing diagnostics")
    _plot_diagnostics(best, y_test, best_scores, best_predictions, best_kind, history)

    payload: dict[str, Any] = {
        "rows": table.to_dict("records"),
        "metrics": list(ALL_METRICS),
        "rank_by": rank_by,
        "strategy": strategy,
        "podium": table.head(3).to_dict("records"),
        "best": best,
        "best_label": REGISTRY[best].label,
        "best_score_kind": best_kind,
        "notes": notes,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        # Carried with the scores so a results table can never be read without
        # knowing which protocol produced it. An ungrouped split scores higher
        # for reasons that have nothing to do with the model.
        "split": split_provenance,
        "embedding_history": history,
        "embedding_params": (
            embedding_settings(**(embedding_params or {}))
            if any(REGISTRY[n].family == "embedding" for n in model_names)
            else None
        ),
    }
    reporter.done("training complete")
    return payload


def _plot_diagnostics(
    name: str,
    y_test: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    kind: str,
    history: list[dict[str, float]] | None,
) -> None:
    """Draw the figures for the winning model.

    Serialised behind a lock because pyplot keeps global state: two jobs
    drawing at once interleave their figures and both come out corrupted.
    """
    from .evaluate import (
        plot_confusion,
        plot_precision_recall,
        plot_threshold_tradeoff,
        plot_training_history,
    )
    from .plotting import FIGURE_LOCK

    label = REGISTRY[name].label
    with FIGURE_LOCK:
        plot_confusion(y_test, predictions, label, filename="ui_confusion_matrix")
        plot_precision_recall({label: (y_test, scores)}, filename="ui_precision_recall")
        plot_threshold_tradeoff(
            y_test, scores, filename="ui_threshold_tradeoff", score_kind=kind
        )
        if history:
            plot_training_history(history, filename="ui_embedding_training")
