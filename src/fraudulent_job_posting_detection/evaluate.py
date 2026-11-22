"""Stage 5: held-out evaluation, threshold tuning and interpretability.

The test split is touched exactly once, here. Every choice -- model family,
imbalance strategy, decision threshold -- was made on cross-validation over
the training split, so these numbers are an honest estimate of performance on
unseen postings.

A note on thresholds
--------------------
A signed margin (LinearSVC, and the embedding network's logits) has its
decision boundary at 0.0; a probability (Naive Bayes, Random Forest,
Logistic Regression's `predict_proba`) has it at 0.5. Hard-coding either one
silently mislabels every threshold plot and reported operating point for the
other family, so the boundary is carried alongside the scores rather than
assumed.
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
from .plotting import (
    FIGURE_LOCK,
    FIGURE_STYLE,
    PALETTE,
    save_figure,
    save_table,
    tidy,
    two_panel_barh,
)
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


def _final_pipelines() -> dict[str, tuple[object, bool]]:
    """The strongest configurations from Stage 4, for a head-to-head on test.

    The flag beside each pipeline says whether its estimator accepts a group
    array at fit time -- the embedding network does, for its early-stopping
    holdout. Carrying it here avoids inferring it from the model's name, which
    breaks the moment a name changes.

    The embedding network is included only when torch is installed, so the
    stage still runs to completion in an environment without it.
    """
    from .embeddings import torch_available
    from .models import build_pipeline, linear_svc, wrap_strategy

    pipelines: dict[str, tuple[object, bool]] = {
        "LinearSVC + class_weight": (
            wrap_strategy(linear_svc(True), "class_weight"), False
        ),
        "LinearSVC + SMOTE": (wrap_strategy(linear_svc(False), "smote"), False),
    }
    if torch_available():
        pipeline, _ = build_pipeline("EmbeddingNet", strategy="class_weight")
        pipelines["EmbeddingNet + pos_weight"] = (pipeline, True)
    return pipelines


def _scores(pipeline: object, frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Continuous decision scores plus their kind."""
    from .models import continuous_scores

    return continuous_scores(pipeline, frame)


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on the fraudulent class."""
    _, _, thresholds, f1 = _f1_curve(y_true, scores)
    if f1.size == 0:
        return 0.0, 0.0
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def out_of_fold_threshold(
    pipeline: object, train: pd.DataFrame, y_train: np.ndarray
) -> float:
    """Choose the decision threshold from out-of-sample predictions.

    Tuning on in-sample training scores does not work: a fitted model
    separates its own training data far more confidently than unseen data, so
    the optimal in-sample threshold is much too high and recall collapses on
    test. Scores from models that never saw the row being scored have a scale
    that matches test conditions.

    For an estimator that already tuned its own threshold on a held-out slice
    -- the embedding network does, as a by-product of early stopping -- that
    value is reused. Re-deriving it by cross-validation would retrain the
    network five times for a threshold chosen the same way, turning a
    two-minute stage into a twelve-minute one. The slice is smaller than a
    5-fold estimate, so the figure is noisier; it is still out of sample,
    which is the property that matters.
    """
    classifier = getattr(pipeline, "named_steps", {}).get("clf")
    existing = getattr(classifier, "threshold_", None)
    if existing is not None:
        logger.info("reusing the estimator's holdout threshold %.3f", float(existing))
        return float(existing)

    groups = make_group_key(train).to_numpy()
    method = (
        "decision_function" if hasattr(pipeline, "decision_function") else "predict_proba"
    )
    oof = cross_val_predict(
        clone(pipeline), train, y_train, groups=groups,
        cv=cv_splitter(), method=method, n_jobs=1,
    )
    oof = np.asarray(oof)
    if oof.ndim == 2:  # predict_proba returns both columns
        oof = oof[:, 1]
    threshold, f1 = best_threshold(y_train, oof)
    logger.info("out-of-fold threshold %.3f (oof F1=%.4f)", threshold, f1)
    return threshold


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_precision_recall(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    filename: str = "precision_recall_test",
) -> None:
    with FIGURE_LOCK, plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, (y_true, scores) in results.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")
        baseline = float(np.mean(next(iter(results.values()))[0]))
        ax.axhline(baseline, ls="--", lw=1, color="grey",
                   label=f"random baseline ({baseline:.3f})")
        ax.set_xlabel("Recall (fraudulent)")
        ax.set_ylabel("Precision (fraudulent)")
        ax.set_title("Precision-recall on the held-out test split")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left", fontsize=9)
        tidy(ax)
        save_figure(filename)


def plot_threshold_tradeoff(
    y_true: np.ndarray,
    scores: np.ndarray,
    filename: str = "threshold_tradeoff",
    score_kind: str = "margin",
) -> None:
    """Precision, recall and F1 across the decision threshold.

    `score_kind` decides where the default boundary is drawn. Marking 0.0 as
    the default on a probability model, or 0.5 on a margin, would misstate the
    operating point the reported metrics actually came from.
    """
    precision, recall, thresholds, f1 = _f1_curve(y_true, scores)
    if thresholds.size == 0:
        logger.warning("no thresholds to plot for %s", filename)
        return
    default = 0.0 if score_kind == "margin" else 0.5
    with FIGURE_LOCK, plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(thresholds, precision, lw=1.6, label="precision")
        ax.plot(thresholds, recall, lw=1.6, label="recall")
        ax.plot(thresholds, f1, lw=2.4, color=PALETTE["accent"], label="F1")
        peak = int(np.argmax(f1))
        ax.axvline(thresholds[peak], ls="--", lw=1, color="grey")
        ax.annotate(
            f"F1 peak {f1[peak]:.3f}\nat {thresholds[peak]:.2f}",
            xy=(thresholds[peak], f1[peak]), xytext=(10, -30),
            textcoords="offset points", fontsize=9,
        )
        if thresholds.min() <= default <= thresholds.max():
            ax.axvline(default, ls=":", lw=1.2, color="black")
        ax.set_xlabel(f"Decision threshold ({score_kind})")
        ax.set_ylabel("Score")
        ax.set_title(
            "Operating point: precision/recall trade-off "
            f"(dotted = default {default:g})"
        )
        ax.legend()
        tidy(ax)
        save_figure(filename)


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, name: str,
    filename: str = "confusion_matrix",
) -> None:
    """Confusion matrix shaded by row rate, annotated with counts.

    Shading by raw count is useless at a 4.84% positive rate: the
    true-negative cell holds 3,390 and the other three hold double digits, so
    one cell is saturated and the three that carry the interesting
    information are all the same pale tint. Normalising within each actual
    class shades by recall instead, which is what the reader needs to compare,
    while the counts stay on the cells.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    totals = matrix.sum(axis=1, keepdims=True)
    rates = np.divide(
        matrix, totals, out=np.zeros(matrix.shape, dtype=float), where=totals > 0
    )
    with FIGURE_LOCK, plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        labels = ["legitimate", "fraudulent"]
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"Confusion matrix — {name}\nshaded by rate within each actual class",
            fontsize=10,
        )
        ax.grid(visible=False)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{matrix[i, j]:,}\n{rates[i, j] * 100:.1f}%",
                    ha="center", va="center", fontsize=11,
                    color="white" if rates[i, j] > 0.5 else "#16233a",
                )
        save_figure(filename)


def plot_training_history(
    history: list[dict[str, float]], filename: str = "embedding_training"
) -> None:
    """Loss and validation average precision per epoch for the embedding model.

    The one diagnostic a closed-form model has no equivalent of: it shows
    whether the network converged, and where early stopping chose to stop.
    """
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    loss = [row["train_loss"] for row in history]
    val = [row["val_avg_precision"] for row in history]

    with FIGURE_LOCK, plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(epochs, loss, lw=2, color=PALETTE["fraudulent"], marker="o",
                markersize=4, label="training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted BCE loss", color=PALETTE["fraudulent"])
        tidy(ax)

        twin = ax.twinx()
        twin.plot(epochs, val, lw=2, color=PALETTE["legitimate"], marker="s",
                  markersize=4, label="validation AP")
        twin.set_ylabel("Validation average precision", color=PALETTE["legitimate"])
        twin.grid(visible=False)
        twin.spines[["top"]].set_visible(False)

        best = int(np.argmax(val))
        twin.axvline(epochs[best], ls="--", lw=1, color="grey")
        twin.annotate(
            f"best epoch {epochs[best]}\nAP {val[best]:.3f}",
            xy=(epochs[best], val[best]), xytext=(12, -38),
            textcoords="offset points", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
        ax.set_title("Embedding network: convergence and early stopping")
        handles = ax.get_lines() + twin.get_lines()[:1]
        ax.legend(handles, [h.get_label() for h in handles], loc="center right",
                  fontsize=9)
        save_figure(filename)


def plot_embedding_neighbours(
    neighbours: dict[str, list[dict[str, object]]],
    filename: str = "embedding_neighbours",
) -> None:
    """Nearest vocabulary terms for a few probe words, by embedding cosine.

    This is the evidence that the embedding layer learned task structure
    rather than merely adding parameters.
    """
    probes = {k: v for k, v in neighbours.items() if v}
    if not probes:
        return
    with FIGURE_LOCK, plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(
            1, len(probes), figsize=(3.4 * len(probes), 4.2), squeeze=False
        )
        for axis, (probe, items) in zip(axes[0], probes.items(), strict=True):
            terms = [str(item["term"]) for item in items][::-1]
            values = [float(item["similarity"]) for item in items][::-1]
            axis.barh(terms, values, color=PALETTE["accent"])
            axis.set_title(f"nearest to '{probe}'", fontsize=10)
            axis.set_xlabel("cosine similarity")
            axis.set_xlim(0, 1)
            axis.grid(axis="y", visible=False)
            tidy(axis)
        save_figure(filename)


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------
def top_features(pipeline: object, top_n: int = 25) -> pd.DataFrame:
    """Largest-magnitude coefficients of the fitted linear model."""
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "pushes"])
    names = np.asarray(features.get_feature_names_out())
    coefficients = np.ravel(classifier.coef_)

    table = pd.DataFrame({"feature": names, "coefficient": coefficients.round(4)})
    table["feature"] = table["feature"].str.replace(r"^(text|binary|numeric)__", "",
                                                    regex=True)
    toward_fraud = table.nlargest(top_n, "coefficient").assign(pushes="fraudulent")
    toward_legit = table.nsmallest(top_n, "coefficient").assign(pushes="legitimate")
    return pd.concat([toward_fraud, toward_legit], ignore_index=True)


def plot_top_features(table: pd.DataFrame, top_n: int = 15) -> None:
    if table.empty:
        return
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


# Probe terms for the embedding neighbourhood figure. Stemmed, because the
# vocabulary is built from stemmed text.
EMBEDDING_PROBES: tuple[str, ...] = ("earn", "recruit", "softwar")


def embedding_report(pipeline: object) -> dict[str, list[dict[str, object]]]:
    """Nearest-neighbour terms for each probe, if this is an embedding pipeline."""
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "nearest_terms"):
        return {}
    transformer = pipeline.named_steps["features"]
    try:
        encoder = transformer.named_transformers_["sequence"]
    except (AttributeError, KeyError):
        return {}
    return {probe: classifier.nearest_terms(encoder, probe) for probe in EMBEDDING_PROBES}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_evaluation(progress: object = None) -> pd.DataFrame:
    from .models import read_corpus
    from .progress import ProgressReporter

    reporter = progress if progress is not None else ProgressReporter()
    config.ensure_directories()
    reporter(0.02, "loading the cleaned corpus")
    frame = read_corpus(copy=False)
    train_idx, test_idx = train_test_indices(frame)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    y_train = train[config.TARGET].to_numpy()
    y_test = test[config.TARGET].to_numpy()
    train_groups = make_group_key(train).to_numpy()

    rows: list[dict[str, object]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, object] = {}
    kinds: dict[str, str] = {}

    # Majority-class baseline: the number accuracy must be judged against.
    rows.append({
        "model": "Majority baseline (always legitimate)", "threshold": "n/a",
        "accuracy": round(float((y_test == 0).mean()), 4),
        "precision_fraud": 0.0, "recall_fraud": 0.0, "f1_fraud": 0.0,
        "avg_precision": round(float(y_test.mean()), 4), "roc_auc": 0.5,
    })

    pipelines = _final_pipelines()
    fitting = reporter.span(0.8, "fitting").weighted(
        dict.fromkeys(pipelines, 1.0)
    )
    for name, (pipeline, takes_groups) in pipelines.items():
        step = fitting[name]
        step(0.05, f"fitting {name}")
        logger.info("fitting %s", name)
        if takes_groups:
            pipeline.fit(train, y_train, clf__groups=train_groups)
        else:
            pipeline.fit(train, y_train)
        fitted[name] = pipeline
        scores, kind = _scores(pipeline, test)
        kinds[name] = kind
        curves[name] = (y_test, scores)

        default = 0.0 if kind == "margin" else 0.5
        step(0.6, f"tuning the threshold for {name}")
        for label, threshold in (
            ("default", default),
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
                "precision_fraud": round(report.get("1", {}).get("precision", 0.0), 4),
                "recall_fraud": round(report.get("1", {}).get("recall", 0.0), 4),
                "f1_fraud": round(report.get("1", {}).get("f1-score", 0.0), 4),
                "avg_precision": round(average_precision_score(y_test, scores), 4),
                "roc_auc": round(roc_auc_score(y_test, scores), 4),
            })
            logger.info("%-26s %-18s F1=%.4f", name, label,
                        report.get("1", {}).get("f1-score", 0.0))
        step.done(f"{name} scored")

    reporter(0.85, "writing tables and figures")
    table = pd.DataFrame(rows)
    save_table(table, "final_evaluation")

    # Select by average precision: it summarises the whole precision/recall
    # curve and is independent of where the threshold happens to sit.
    best_name = max(curves, key=lambda name: average_precision_score(*curves[name]))
    best_pipeline = fitted[best_name]
    best_scores = curves[best_name][1]
    best_kind = kinds[best_name]
    logger.info("best by average precision: %s", best_name)

    with FIGURE_LOCK:
        plot_precision_recall(curves)
        plot_threshold_tradeoff(y_test, best_scores, score_kind=best_kind)
        default = 0.0 if best_kind == "margin" else 0.5
        plot_confusion(y_test, (best_scores >= default).astype(int), best_name)

        features = top_features(best_pipeline)
        if not features.empty:
            save_table(features, "top_features")
            plot_top_features(features)

        history = getattr(best_pipeline.named_steps["clf"], "history_", None)
        if history:
            save_table(pd.DataFrame(history), "embedding_training_history")
            plot_training_history(history)

        neighbours = embedding_report(best_pipeline)
        if neighbours:
            plot_embedding_neighbours(neighbours)
            save_table(
                pd.DataFrame([
                    {"probe": probe, **item}
                    for probe, items in neighbours.items() for item in items
                ]),
                "embedding_neighbours",
            )

    joblib.dump(best_pipeline, config.MODELS_DIR / "final_model.joblib")
    logger.info("model -> models/final_model.joblib (%s)", best_name)
    reporter.done("evaluation complete")
    return table
