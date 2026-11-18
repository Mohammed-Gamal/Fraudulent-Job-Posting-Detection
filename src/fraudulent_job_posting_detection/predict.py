"""Scoring a single pasted job advert, with a term-level explanation.

The posting is cleaned through `build_single_posting`, which calls the same
`build_corpus` used for the training data. Reimplementing the cleaning here
would be the classic source of train/serve skew: the model would receive
features shaped differently from those it learned on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .preprocess import DEFAULT_OPTIONS, build_single_posting, read_corpus_meta

logger = logging.getLogger(__name__)


def corpus_options() -> dict[str, bool]:
    """Cleaning options the stored corpus was built with, falling back to defaults."""
    meta = read_corpus_meta()
    if meta and isinstance(meta.get("options"), dict):
        return {**DEFAULT_OPTIONS, **meta["options"]}
    return dict(DEFAULT_OPTIONS)


def explain(pipeline: object, row: pd.DataFrame, top_n: int = 12) -> list[dict[str, object]]:
    """Per-feature contributions for a linear model: coefficient x feature value.

    The coefficient alone is not the contribution. A term with a large weight
    that barely appears contributes little, and the TF-IDF and binary blocks
    differ in scale by roughly 14x, so only the product is comparable across
    them.

    Returns [] for estimators without linear coefficients, such as the forest.
    """
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return []

    vector = features.transform(row)
    coefficients = np.ravel(classifier.coef_)
    names = np.asarray(features.get_feature_names_out())

    dense = np.asarray(vector.todense()).ravel() if hasattr(vector, "todense") else np.ravel(vector)
    contributions = dense * coefficients

    active = np.nonzero(dense)[0]
    if active.size == 0:
        return []
    ranked = active[np.argsort(-np.abs(contributions[active]))][:top_n]

    return [
        {
            "feature": str(names[i]).split("__", 1)[-1],
            "value": round(float(dense[i]), 4),
            "coefficient": round(float(coefficients[i]), 4),
            "contribution": round(float(contributions[i]), 4),
            "pushes": "fraudulent" if contributions[i] > 0 else "legitimate",
        }
        for i in ranked
    ]


def predict_posting(
    pipeline: object, values: dict[str, object], options: dict[str, bool] | None = None
) -> dict[str, object]:
    """Classify one posting and report what drove the decision."""
    settings = options if options is not None else corpus_options()
    row = build_single_posting(values, **settings)

    label = int(pipeline.predict(row)[0])

    if hasattr(pipeline, "decision_function"):
        raw = float(pipeline.decision_function(row)[0])
        score_kind, score = "margin", raw
    else:
        score = float(pipeline.predict_proba(row)[0, 1])
        score_kind = "probability"

    return {
        "label": "FRAUDULENT" if label else "LEGITIMATE",
        "is_fraud": bool(label),
        "score": round(score, 4),
        "score_kind": score_kind,
        "cleaned_text": str(row.loc[0, "text_all"]),
        "n_tokens": int(row.loc[0, "n_tokens"]),
        "contributions": explain(pipeline, row),
    }
