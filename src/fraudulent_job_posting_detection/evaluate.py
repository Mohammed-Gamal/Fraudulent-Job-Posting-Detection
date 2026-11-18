"""Stage 5: held-out evaluation, threshold tuning and interpretability.

The test split is touched exactly once, here. Every choice -- model family,
imbalance strategy, decision threshold -- was made on cross-validation over
the training split, so these numbers are an honest estimate of performance
on unseen postings.
"""

from __future__ import annotations

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from . import config
from .plotting import save_figure, save_table, two_panel_barh
from .splits import cv_splitter, make_group_key, train_test_indices

logger = logging.getLogger(__name__)


def _f1_curve(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall, thresholds and F1 at every threshold.

    precision_recall_curve returns one more point than it does thresholds, so
    the final element is dropped to keep the four arrays aligned.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    return precision, recall, thresholds, f1


def _final_pipelines() -> dict[str, object]:
    """The two strongest configurations from Stage 4, for a head-to-head on test."""
    from .models import linear_svc, wrap_strategy

    return {
        "LinearSVC + class_weight": wrap_strategy(linear_svc(True), "class_weight"),
        "LinearSVC + SMOTE": wrap_strategy(linear_svc(False), "smote"),
    }


def _scores(pipeline: object, frame: pd.DataFrame) -> np.ndarray:
    """Continuous decision scores, whichever interface the estimator exposes."""
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(frame)
    return pipeline.predict_proba(frame)[:, 1]


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on the fraudulent class."""
    _, _, thresholds, f1 = _f1_curve(y_true, scores)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])



def out_of_fold_threshold(
    pipeline: object, train: pd.DataFrame, y_train: np.ndarray
) -> float:
    """Choose the decision threshold from out-of-fold predictions.

    Tuning on in-sample training scores does not work: a fitted model
    separates its own training data far more confidently than unseen data, so
    the optimal in-sample threshold is much too high and recall collapses on
    test. Out-of-fold scores come from models that never saw the row being
    scored, so their scale matches test conditions.
    """
    groups = make_group_key(train).to_numpy()
    oof = cross_val_predict(
        clone(pipeline), train, y_train, groups=groups,
        cv=cv_splitter(), method="decision_function", n_jobs=1,
    )
    threshold, f1 = best_threshold(y_train, oof)
    logger.info("out-of-fold threshold %.3f (oof F1=%.4f)", threshold, f1)
    return threshold


def plot_precision_recall(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    filename: str = "precision_recall_test",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y_true, scores) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    baseline = float(np.mean(next(iter(results.values()))[0]))
    ax.axhline(baseline, ls="--", lw=1, color="grey",
               label=f"random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (fraudulent)")
    ax.set_ylabel("Precision (fraudulent)")
    ax.set_title("Precision-recall on the held-out test split")
    ax.legend(loc="lower left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(filename)


def plot_threshold_tradeoff(
    y_true: np.ndarray, scores: np.ndarray,
    filename: str = "threshold_tradeoff",
) -> None:
    precision, recall, thresholds, f1 = _f1_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thresholds, precision, label="precision")
    ax.plot(thresholds, recall, label="recall")
    ax.plot(thresholds, f1, label="F1", lw=2)
    peak = int(np.argmax(f1))
    ax.axvline(thresholds[peak], ls="--", lw=1, color="grey")
    ax.annotate(
        f"F1 peak {f1[peak]:.3f}\nat {thresholds[peak]:.2f}",
        xy=(thresholds[peak], f1[peak]), xytext=(10, -30),
        textcoords="offset points", fontsize=9,
    )
    ax.axvline(0.0, ls=":", lw=1, color="black")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Operating point: precision/recall trade-off (dotted = default 0.0)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(filename)


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, name: str,
    filename: str = "confusion_matrix",
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap="Blues")
    labels = ["legitimate", "fraudulent"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix\n{name}", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    save_figure(filename)


def top_features(pipeline: object, top_n: int = 25) -> pd.DataFrame:
    """Largest-magnitude coefficients of the fitted linear model."""
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["clf"]
    names = np.asarray(features.get_feature_names_out())
    coefficients = np.ravel(classifier.coef_)

    table = pd.DataFrame({"feature": names, "coefficient": coefficients.round(4)})
    table["feature"] = table["feature"].str.replace(r"^(text|binary|numeric)__", "",
                                                    regex=True)
    toward_fraud = table.nlargest(top_n, "coefficient").assign(pushes="fraudulent")
    toward_legit = table.nsmallest(top_n, "coefficient").assign(pushes="legitimate")
    return pd.concat([toward_fraud, toward_legit], ignore_index=True)


def plot_top_features(table: pd.DataFrame, top_n: int = 15) -> None:
    two_panel_barh(
        table[table["pushes"] == "fraudulent"].nlargest(top_n, "coefficient").iloc[::-1],
        table[table["pushes"] == "legitimate"].nsmallest(top_n, "coefficient"),
        value_column="coefficient",
        label_column="feature",
        xlabel="LinearSVC coefficient",
        filename="top_features",
        titles=("Pushes toward FRAUDULENT", "Pushes toward LEGITIMATE"),
        max_label=24,
    )


def run_evaluation() -> pd.DataFrame:
    config.ensure_directories()
    frame = pd.read_parquet(config.PROCESSED_DIR / "corpus.parquet")
    train_idx, test_idx = train_test_indices(frame)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    y_train = train[config.TARGET].to_numpy()
    y_test = test[config.TARGET].to_numpy()

    rows: list[dict[str, object]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, object] = {}

    # Majority-class baseline: the number accuracy must be judged against.
    rows.append({
        "model": "Majority baseline (always legitimate)", "threshold": "n/a",
        "accuracy": round(float((y_test == 0).mean()), 4),
        "precision_fraud": 0.0, "recall_fraud": 0.0, "f1_fraud": 0.0,
        "avg_precision": round(float(y_test.mean()), 4), "roc_auc": 0.5,
    })

    for name, pipeline in _final_pipelines().items():
        pipeline.fit(train, y_train)
        fitted[name] = pipeline
        scores = _scores(pipeline, test)
        curves[name] = (y_test, scores)

        for label, threshold in (
            ("default", 0.0),
            ("tuned", out_of_fold_threshold(pipeline, train, y_train)),
        ):
            predictions = (scores >= threshold).astype(int)
            report = classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
            rows.append({
                "model": name,
                "threshold": f"{label} ({threshold:.3f})",
                "accuracy": round(report["accuracy"], 4),
                "precision_fraud": round(report["1"]["precision"], 4),
                "recall_fraud": round(report["1"]["recall"], 4),
                "f1_fraud": round(report["1"]["f1-score"], 4),
                "avg_precision": round(average_precision_score(y_test, scores), 4),
                "roc_auc": round(roc_auc_score(y_test, scores), 4),
            })
            logger.info("%-26s %-18s F1=%.4f", name, label, report["1"]["f1-score"])

    table = pd.DataFrame(rows)
    save_table(table, "final_evaluation")

    # Select by average precision: it summarises the whole precision/recall
    # curve and is independent of where the threshold happens to sit.
    best_name = max(
        curves, key=lambda name: average_precision_score(*curves[name])
    )
    best_pipeline = fitted[best_name]
    best_scores = curves[best_name][1]
    logger.info("best by average precision: %s", best_name)

    plot_precision_recall(curves)
    plot_threshold_tradeoff(y_test, best_scores)
    plot_confusion(y_test, (best_scores >= 0.0).astype(int), best_name)

    if hasattr(best_pipeline.named_steps["clf"], "coef_"):
        features = top_features(best_pipeline)
        save_table(features, "top_features")
        plot_top_features(features)

    joblib.dump(best_pipeline, config.MODELS_DIR / "final_model.joblib")
    logger.info("model -> models/final_model.joblib (%s)", best_name)
    return table
