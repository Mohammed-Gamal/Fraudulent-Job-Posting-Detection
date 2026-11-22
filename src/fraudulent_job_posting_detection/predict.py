"""Scoring a single pasted job advert, with an explanation of the decision.

The posting is cleaned through `build_single_posting`, which calls the same
`build_corpus` used for the training data. Reimplementing the cleaning here
would be the classic source of train/serve skew: the model would receive
features shaped differently from those it learned on.

Three kinds of explanation, because no single one covers every model:

coefficients
    For linear models, contribution = feature value x weight. Exact, and the
    most informative when available.

attention
    For the embedding network, the pooling weights over token positions:
    which words the model actually looked at. Its answer to coefficients.

sensitivity
    Toggle one structured flag, re-score, report the change. Model-agnostic,
    so Random Forest -- which has neither coefficients nor attention -- is no
    longer a black box in the interface. It also answers the question a user
    actually asks: "what if this posting had a company logo?"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .preprocess import DEFAULT_OPTIONS, build_single_posting, read_corpus_meta

logger = logging.getLogger(__name__)

FLAG_LABELS: dict[str, str] = {
    "telecommuting": "Telecommuting",
    "has_company_logo": "Has company logo",
    "has_questions": "Has screening questions",
}


def corpus_options() -> dict[str, bool]:
    """Cleaning options the stored corpus was built with, falling back to defaults."""
    meta = read_corpus_meta()
    if meta and isinstance(meta.get("options"), dict):
        return {**DEFAULT_OPTIONS, **meta["options"]}
    return dict(DEFAULT_OPTIONS)


def _dense(vector: Any) -> np.ndarray:
    if hasattr(vector, "toarray"):
        return np.asarray(vector.toarray()).ravel()
    return np.asarray(vector).ravel()


# ---------------------------------------------------------------------------
# Linear models: coefficient x value
# ---------------------------------------------------------------------------
def explain(pipeline: Any, row: pd.DataFrame, top_n: int = 12) -> list[dict[str, Any]]:
    """Per-feature contributions for a linear model: coefficient x feature value.

    The coefficient alone is not the contribution. A term with a large weight
    that barely appears contributes little, and the TF-IDF and binary blocks
    differ in scale by roughly 14x, so only the product is comparable across
    them.

    Returns [] for estimators without linear coefficients.
    """
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return []

    vector = _dense(features.transform(row))
    coefficients = np.ravel(classifier.coef_)
    names = np.asarray(features.get_feature_names_out())
    if coefficients.shape[0] != vector.shape[0]:
        logger.warning(
            "coefficient/feature length mismatch (%s vs %s); skipping explanation",
            coefficients.shape[0], vector.shape[0],
        )
        return []

    contributions = vector * coefficients
    active = np.nonzero(vector)[0]
    if active.size == 0:
        return []
    ranked = active[np.argsort(-np.abs(contributions[active]))][:top_n]

    return [
        {
            "feature": str(names[i]).split("__", 1)[-1],
            "value": round(float(vector[i]), 4),
            "coefficient": round(float(coefficients[i]), 4),
            "contribution": round(float(contributions[i]), 4),
            "pushes": "fraudulent" if contributions[i] > 0 else "legitimate",
        }
        for i in ranked
    ]


# ---------------------------------------------------------------------------
# Embedding model: attention over tokens
# ---------------------------------------------------------------------------
def _sequence_encoder(pipeline: Any) -> Any | None:
    try:
        return pipeline.named_steps["features"].named_transformers_["sequence"]
    except (AttributeError, KeyError):
        return None


def explain_attention(
    pipeline: Any, row: pd.DataFrame, top_n: int = 12
) -> list[dict[str, Any]]:
    """Tokens the embedding network attended to, highest weight first.

    Weights for repeats of the same token are summed, because a term appearing
    three times with moderate attention each is genuinely carrying more of the
    decision than a single moderate hit.
    """
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "attention_weights"):
        return []
    encoder = _sequence_encoder(pipeline)
    if encoder is None:
        return []

    features = pipeline.named_steps["features"]
    matrix = np.asarray(features.transform(row))
    try:
        weights = classifier.attention_weights(matrix)[0]
    except Exception:
        logger.exception("attention extraction failed")
        return []

    from .embeddings import PAD_ID

    sequence_length = int(getattr(encoder, "sequence_length", weights.shape[0]))
    ids = matrix[0, :sequence_length].astype(int)
    inverse = {index: token for token, index in encoder.vocabulary_.items()}

    totals: dict[str, dict[str, float]] = {}
    for position, token_id in enumerate(ids):
        if token_id == PAD_ID or position >= weights.shape[0]:
            continue
        token = inverse.get(int(token_id), "<unk>")
        entry = totals.setdefault(token, {"weight": 0.0, "count": 0.0})
        entry["weight"] += float(weights[position])
        entry["count"] += 1.0

    ranked = sorted(totals.items(), key=lambda item: -item[1]["weight"])[:top_n]
    total_weight = sum(entry["weight"] for entry in totals.values()) or 1.0
    return [
        {
            "token": token,
            "attention": round(entry["weight"], 5),
            "share": round(entry["weight"] / total_weight * 100, 2),
            "occurrences": int(entry["count"]),
        }
        for token, entry in ranked
    ]


def neighbour_terms(
    pipeline: Any, tokens: list[str], top_n: int = 5
) -> list[dict[str, Any]]:
    """For each token, its nearest neighbours in the learned embedding space.

    Deliberately *not* shown in the per-prediction panel. Inspecting the
    trained vectors on this corpus showed the neighbourhoods are not
    semantically interpretable -- `recruit` sits beside `class` and
    `southwest` -- because 14,000 documents with 866 positives is far too
    little signal to shape 1.3M embedding parameters into a similarity space,
    and because the objective rewards class separation rather than similarity.

    Presenting those lists beside a verdict would imply the model had learned
    word meaning, which it has not. It is kept here because the negative
    result is worth reporting: `fjpd evaluate` writes it to
    reports/tables/embedding_neighbours.csv as a finding.
    """
    classifier = pipeline.named_steps["clf"]
    encoder = _sequence_encoder(pipeline)
    if encoder is None or not hasattr(classifier, "nearest_terms"):
        return []
    results: list[dict[str, Any]] = []
    for token in tokens:
        neighbours = classifier.nearest_terms(encoder, token, top_n=top_n)
        if neighbours:
            results.append({"term": token, "neighbours": neighbours})
    return results


# ---------------------------------------------------------------------------
# Any model: toggle a flag and re-score
# ---------------------------------------------------------------------------
def flag_sensitivity(
    pipeline: Any, values: dict[str, Any], options: dict[str, bool]
) -> list[dict[str, Any]]:
    """Change in score when each structured flag is flipped.

    Works for every model family, including the Random Forest, which exposes
    neither coefficients nor attention. One extra forward pass per flag --
    three in total -- so it is cheap enough to run on every prediction.
    """
    from .models import continuous_scores

    baseline_row = build_single_posting(values, **options)
    baseline, kind = continuous_scores(pipeline, baseline_row)
    baseline_score = float(np.ravel(baseline)[0])

    rows: list[dict[str, Any]] = []
    for flag in config.BINARY_FLAGS:
        flipped = dict(values)
        current = int(bool(values.get(flag, 0)))
        flipped[flag] = 1 - current
        try:
            scores, _ = continuous_scores(
                pipeline, build_single_posting(flipped, **options)
            )
            alternative = float(np.ravel(scores)[0])
        except Exception:
            logger.exception("sensitivity check failed for %s", flag)
            continue
        delta = alternative - baseline_score
        rows.append({
            "flag": flag,
            "label": FLAG_LABELS.get(flag, flag),
            "current": current,
            "if_flipped": 1 - current,
            "score_if_flipped": round(alternative, 4),
            "delta": round(delta, 4),
            # Which way the score would move, in the user's terms.
            "direction": (
                "more suspicious" if delta > 1e-6
                else "less suspicious" if delta < -1e-6
                else "no change"
            ),
        })
    rows.sort(key=lambda row: -abs(row["delta"]))
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def predict_posting(
    pipeline: Any,
    values: dict[str, Any],
    options: dict[str, bool] | None = None,
    include_sensitivity: bool = True,
) -> dict[str, Any]:
    """Classify one posting and report what drove the decision."""
    from .models import continuous_scores, operating_threshold

    settings = options if options is not None else corpus_options()
    row = build_single_posting(values, **settings)

    scores, kind = continuous_scores(pipeline, row)
    score = float(np.ravel(scores)[0])
    # The model's own operating point, not the family default: the embedding
    # network tunes its cut on a holdout, and showing the verdict against 0.0
    # while it predicts against 3.9 would contradict its own label.
    threshold = operating_threshold(pipeline)
    is_fraud = score >= threshold

    # A readable 0-1 figure beside the raw score, centred on the decision
    # boundary so 0.5 means "exactly on the line" whatever the threshold is.
    # Monotonic in the score, so the ordering is preserved -- but it is a
    # squashed margin, not a calibrated probability, and is labelled as such
    # in the interface.
    confidence = (
        1.0 / (1.0 + np.exp(-(score - threshold))) if kind == "margin" else score
    )

    attention = explain_attention(pipeline, row)
    result: dict[str, Any] = {
        "label": "FRAUDULENT" if is_fraud else "LEGITIMATE",
        "is_fraud": bool(is_fraud),
        "score": round(score, 4),
        "score_kind": kind,
        "threshold": threshold,
        "confidence": round(float(confidence), 4),
        "cleaned_text": str(row.loc[0, "text_all"]),
        "n_tokens": int(row.loc[0, "n_tokens"]),
        "contributions": explain(pipeline, row),
        "attention": attention,
        "sensitivity": (
            flag_sensitivity(pipeline, values, settings) if include_sensitivity else []
        ),
    }
    return result
