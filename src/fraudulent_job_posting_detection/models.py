"""Stages 3-4: feature extraction and model comparison.

Everything is expressed as a scikit-learn Pipeline so that each transform is
fitted inside the cross-validation fold. Vectorising the whole corpus before
splitting would let test-set vocabulary and document frequencies inform
training -- a subtler leak than duplicate rows, and just as damaging.

All feature blocks are non-negative, which keeps Multinomial Naive Bayes
applicable alongside the discriminative models.
"""

from __future__ import annotations

import logging

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
from .splits import cv_splitter, make_group_key

logger = logging.getLogger(__name__)

TEXT_COLUMN = "text_all"

# Structured features: the binary flags plus the missingness indicators that
# Stage 1 showed to be strongly associated with fraud.
STRUCTURED_BINARY: list[str] = config.BINARY_FLAGS + [
    f"{field}_is_missing"
    for field in config.TEXT_FIELDS + config.CATEGORICAL_FIELDS
]
STRUCTURED_NUMERIC: list[str] = ["n_tokens"]

SCORING = {
    "f1_fraud": "f1",
    "precision_fraud": "precision",
    "recall_fraud": "recall",
    "avg_precision": "average_precision",
    "roc_auc": "roc_auc",
}


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
    blocks: list[tuple[str, object, object]] = [
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
        blocks.append(("binary", "passthrough", STRUCTURED_BINARY))
        blocks.append(("numeric", MinMaxScaler(), STRUCTURED_NUMERIC))
    return ColumnTransformer(blocks, sparse_threshold=1.0)


def candidate_models(balanced: bool = True) -> dict[str, object]:
    """The four classifiers compared in the report.

    `balanced` applies inverse-frequency class weighting where the estimator
    supports it. Multinomial NB has no such parameter, which is itself a
    finding worth reporting.
    """
    weight = "balanced" if balanced else None
    return {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight=weight, random_state=config.RANDOM_STATE
        ),
        "LinearSVC": LinearSVC(
            class_weight=weight, random_state=config.RANDOM_STATE,
            dual="auto", max_iter=20_000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight=weight,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def make_pipeline(estimator: object, **feature_kwargs: object) -> Pipeline:
    return Pipeline(
        [("features", build_features(**feature_kwargs)), ("clf", estimator)]
    )


def cross_validate_models(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    models: dict[str, object] | None = None,
    use_structured: bool = True,
) -> pd.DataFrame:
    """Grouped, stratified cross-validation of each candidate on the training set."""
    models = models if models is not None else candidate_models()
    train = frame.iloc[train_idx]
    labels = train[config.TARGET].to_numpy()
    groups = make_group_key(train).to_numpy()

    rows: list[dict[str, object]] = []
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator, use_structured=use_structured)
        # LinearSVC exposes decision_function but not predict_proba; the
        # ranking metrics accept either, so no special casing is needed.
        scores = cross_validate(
            pipeline,
            train,
            labels,
            groups=groups,
            cv=cv_splitter(),
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        row: dict[str, object] = {"model": name}
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


def majority_baseline(frame: pd.DataFrame, test_idx: np.ndarray) -> pd.DataFrame:
    """The 'predict legitimate always' baseline, to put accuracy in context."""
    labels = frame.iloc[test_idx][config.TARGET].to_numpy()
    return pd.DataFrame(
        [
            {
                "model": "MajorityBaseline (always legitimate)",
                "accuracy": round(float((labels == 0).mean()), 4),
                "f1_fraud": 0.0,
                "precision_fraud": 0.0,
                "recall_fraud": 0.0,
            }
        ]
    )


RESULTS_PATH = config.INTERIM_DIR / "cv_results.csv"

FEATURE_SETS: dict[str, bool] = {
    "tfidf + structured": True,
    "tfidf only": False,
}


def _completed() -> set[tuple[str, str]]:
    if not RESULTS_PATH.exists():
        return set()
    done = pd.read_csv(RESULTS_PATH)
    return set(zip(done["model"], done["features"], strict=True))


def run_comparison(only: str | None = None) -> pd.DataFrame:
    """Stage 4a: cross-validate every candidate, resuming where it left off.

    Each completed (model, feature set) pair is appended to disk immediately,
    so an interrupted run loses at most the model in flight.
    """
    from .explore import save_table
    from .splits import train_test_indices

    config.ensure_directories()
    frame = pd.read_parquet(config.PROCESSED_DIR / "corpus.parquet")
    train_idx, _ = train_test_indices(frame)

    done = _completed()
    for feature_label, use_structured in FEATURE_SETS.items():
        for name, estimator in candidate_models().items():
            if only is not None and name != only:
                continue
            if (name, feature_label) in done:
                logger.info("skip %s / %s (already done)", name, feature_label)
                continue
            result = cross_validate_models(
                frame, train_idx, {name: estimator}, use_structured=use_structured
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


def run_imbalance_comparison() -> pd.DataFrame:
    """Stage 4b: does the imbalance remedy matter, and which one?

    Compares three treatments on the strongest classifier:
      - nothing: the estimator sees the raw 20:1 ratio
      - class weighting: errors on the minority class cost proportionally more
      - SMOTE: synthesise minority examples in feature space

    SMOTE is applied *inside* the pipeline, so it is fitted on the training
    folds only. Oversampling before cross-validation would place synthetic
    copies of test examples into training -- a frequent and severe error.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    from .explore import save_table
    from .splits import train_test_indices

    config.ensure_directories()
    frame = pd.read_parquet(config.PROCESSED_DIR / "corpus.parquet")
    train_idx, _ = train_test_indices(frame)
    train = frame.iloc[train_idx]
    labels = train[config.TARGET].to_numpy()
    groups = make_group_key(train).to_numpy()

    def svc(balanced: bool) -> LinearSVC:
        return LinearSVC(
            class_weight="balanced" if balanced else None,
            random_state=config.RANDOM_STATE, dual="auto", max_iter=20_000,
        )

    strategies: dict[str, object] = {
        "none": Pipeline([("features", build_features()), ("clf", svc(False))]),
        "class_weight=balanced": Pipeline(
            [("features", build_features()), ("clf", svc(True))]
        ),
        "SMOTE": ImbPipeline(
            [
                ("features", build_features()),
                ("smote", SMOTE(random_state=config.RANDOM_STATE, k_neighbors=5)),
                ("clf", svc(False)),
            ]
        ),
    }

    rows: list[dict[str, object]] = []
    for name, pipeline in strategies.items():
        scores = cross_validate(
            pipeline, train, labels, groups=groups, cv=cv_splitter(),
            scoring=SCORING, n_jobs=1, error_score="raise",
        )
        row: dict[str, object] = {"strategy": name, "model": "LinearSVC"}
        for metric in SCORING:
            row[metric] = round(float(scores[f"test_{metric}"].mean()), 4)
        rows.append(row)
        logger.info("%-22s F1=%.4f  P=%.4f  R=%.4f",
                    name, row["f1_fraud"], row["precision_fraud"], row["recall_fraud"])

    table = pd.DataFrame(rows)
    save_table(table, "imbalance_comparison")
    return table


# ---------------------------------------------------------------------------
# Single-split training, used by the web interface
# ---------------------------------------------------------------------------
ALL_METRICS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision_fraud": "Precision (fraud)",
    "recall_fraud": "Recall (fraud)",
    "f1_fraud": "F1 (fraud)",
    "avg_precision": "Average precision",
    "roc_auc": "ROC AUC",
}

IMBALANCE_STRATEGIES: tuple[str, ...] = ("none", "class_weight", "smote")


def _wrap_strategy(estimator: object, strategy: str) -> object:
    """Attach the chosen imbalance treatment around a bare estimator."""
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


UI_MODEL_PATH = config.MODELS_DIR / "ui_model.joblib"

# Figures regenerated after every interactive training run, named apart from
# the CLI's so an exploratory run never overwrites report output.
UI_FIGURES: tuple[str, ...] = (
    "ui_confusion_matrix.png",
    "ui_precision_recall.png",
    "ui_threshold_tradeoff.png",
)


def _save_and_plot_best(
    name: str,
    pipeline: object,
    y_test: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """Persist the winning pipeline and draw its diagnostic plots.

    Saved so the prediction page has a model to use: training results that
    vanish when the process restarts are of no use for a demo.
    """
    import joblib

    from .evaluate import plot_confusion, plot_precision_recall, plot_threshold_tradeoff

    joblib.dump({"name": name, "pipeline": pipeline}, UI_MODEL_PATH)
    plot_confusion(y_test, predictions, name, filename="ui_confusion_matrix")
    plot_precision_recall({name: (y_test, scores)}, filename="ui_precision_recall")
    plot_threshold_tradeoff(y_test, scores, filename="ui_threshold_tradeoff")
    logger.info("best model %s saved to %s", name, UI_MODEL_PATH.name)


def load_ui_model() -> tuple[str, object] | None:
    """The pipeline saved by the last interactive training run, if any."""
    if not UI_MODEL_PATH.exists():
        return None
    import joblib

    try:
        payload = joblib.load(UI_MODEL_PATH)
        return payload["name"], payload["pipeline"]
    except Exception:
        logger.exception("could not load %s", UI_MODEL_PATH)
        return None


def train_and_evaluate(
    frame: pd.DataFrame,
    model_names: list[str],
    strategy: str = "class_weight",
    rank_by: str = "f1_fraud",
    progress: object = None,
) -> pd.DataFrame:
    """Fit each selected model on the training split and score it on the test split.

    A single split rather than cross-validation: the GUI needs to stay
    responsive, and one grouped split is enough to rank models. Use
    `fjpd compare` for the cross-validated numbers that belong in a report.

    `progress` is an optional callable taking (done, total, label).
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    from .splits import train_test_indices

    train_idx, test_idx = train_test_indices(frame)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    y_train = train[config.TARGET].to_numpy()
    y_test = test[config.TARGET].to_numpy()

    available = candidate_models(balanced=(strategy == "class_weight"))
    rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for position, name in enumerate(model_names, start=1):
        if progress is not None:
            progress(position - 1, len(model_names), f"training {name}")
        estimator = available[name]
        pipeline = _wrap_strategy(estimator, strategy)
        pipeline.fit(train, y_train)

        predictions = pipeline.predict(test)
        if hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(test)
        else:
            scores = pipeline.predict_proba(test)[:, 1]

        fitted[name] = pipeline
        outputs[name] = (scores, predictions)
        rows.append({
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_fraud": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall_fraud": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1_fraud": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "avg_precision": round(float(average_precision_score(y_test, scores)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        })
        logger.info("%s trained: F1=%.4f", name, rows[-1]["f1_fraud"])

    table = pd.DataFrame(rows).sort_values(rank_by, ascending=False)
    table = table.reset_index(drop=True)

    best = str(table.loc[0, "model"])
    if progress is not None:
        progress(len(model_names), len(model_names), f"saving {best}")
    best_scores, best_predictions = outputs[best]
    _save_and_plot_best(best, fitted[best], y_test, best_scores, best_predictions)

    if progress is not None:
        progress(len(model_names), len(model_names), "done")
    return table
