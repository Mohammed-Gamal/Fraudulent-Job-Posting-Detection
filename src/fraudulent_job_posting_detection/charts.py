"""Exploratory results as data, for the interface to draw.

`explore.py` renders the same analyses to PNG for the report. This module
returns the numbers instead, so the page can draw them as SVG the user can
hover, filter and re-rank. The analysis functions are shared -- a chart here
and the figure in `reports/figures/` come from the same computation, so they
cannot disagree.

Every chart declares what it is, which keeps the knowledge in one place: the
interface builds its checkboxes from `CHART_SPECS` rather than hardcoding a
list that drifts as charts are added.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChartSpec:
    """What a chart is, so the interface can offer it without knowing more."""

    key: str
    label: str
    blurb: str
    kind: str          # histogram | grouped_bar | diverging_bar | paired_bar | ranked_bar
    default: bool = True
    # Charts that take a parameter, e.g. which categorical field to rank.
    options: tuple[str, ...] = ()


CHART_SPECS: tuple[ChartSpec, ...] = (
    ChartSpec(
        key="length",
        label="Description length by class",
        blurb=(
            "Whether fraudulent adverts are systematically shorter. If they "
            "are, length is a cheap feature — and a warning that a model may "
            "key on length rather than content."
        ),
        kind="histogram",
    ),
    ChartSpec(
        key="flags",
        label="Fraud rate by binary flag",
        blurb=(
            "The absence of an institutional footprint is the strongest "
            "signal in the dataset, stronger than any single word."
        ),
        kind="grouped_bar",
    ),
    ChartSpec(
        key="terms",
        label="Distinguishing vocabulary",
        blurb=(
            "Terms most associated with each class by smoothed log-odds, "
            "counted once per posting so one advert repeating a word cannot "
            "dominate."
        ),
        kind="diverging_bar",
        options=("normalised", "raw"),
    ),
    ChartSpec(
        key="missingness",
        label="Missing fields by class",
        blurb=(
            "A field being absent is itself the signal: a posting with no "
            "company profile is a posting with no company behind it."
        ),
        kind="paired_bar",
    ),
    ChartSpec(
        key="category",
        label="Fraud rate by category",
        blurb=(
            "Which industries, functions and contract types carry the most "
            "fraud. Values with fewer than 30 postings are dropped: two of "
            "three reads as a 67% fraud rate and is noise."
        ),
        kind="ranked_bar",
        options=tuple(config.CATEGORICAL_FIELDS),
    ),
)

CHART_BY_KEY: dict[str, ChartSpec] = {spec.key: spec for spec in CHART_SPECS}


def catalogue() -> list[dict[str, Any]]:
    """The chart list the interface renders its checkboxes from."""
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "blurb": spec.blurb,
            "kind": spec.kind,
            "default": spec.default,
            "options": list(spec.options),
        }
        for spec in CHART_SPECS
    ]


def validate_keys(keys: list[str]) -> list[str]:
    """Requested chart keys that exist, in declaration order."""
    wanted = set(keys)
    return [spec.key for spec in CHART_SPECS if spec.key in wanted]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
# Each chart is a full pass over 17,880 rows and the vocabulary one is a
# CountVectorizer fit, which is a second or two. The page re-requests charts
# whenever the user ticks a box, so the results are memoised against the raw
# file's identity and thrown away when the dataset is replaced.
_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple, Any] = {}
_CACHE_KEY: list[Any] = [None]


def _dataset_key() -> tuple[int, int] | None:
    try:
        stat = config.RAW_CSV.stat()
    except OSError:
        return None
    return (stat.st_size, int(stat.st_mtime_ns))


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_KEY[0] = None


def _cached(name: tuple, build) -> Any:
    key = _dataset_key()
    with _CACHE_LOCK:
        if _CACHE_KEY[0] != key:
            _CACHE.clear()
            _CACHE_KEY[0] = key
        if name in _CACHE:
            return _CACHE[name]
    value = build()
    with _CACHE_LOCK:
        _CACHE[name] = value
    return value


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _length_chart(frame: pd.DataFrame, bins: int = 40) -> dict[str, Any]:
    """Description word counts per class, as shared-bin densities.

    Densities rather than counts: with 19 legitimate postings for every
    fraudulent one, raw counts would render the fraudulent series as a flat
    line along the axis and say nothing about its shape.
    """
    words = frame["description"].fillna("").astype(str).str.split().str.len()
    upper = float(words.quantile(0.99)) or 1.0
    edges = np.linspace(0, upper, bins + 1)

    series = []
    for label, mask in (
        ("legitimate", frame[config.TARGET] == 0),
        ("fraudulent", frame[config.TARGET] == 1),
    ):
        subset = words[mask]
        counts, _ = np.histogram(subset, bins=edges)
        total = counts.sum() or 1
        series.append({
            "label": label,
            "counts": counts.astype(int).tolist(),
            "density": (counts / total).round(5).tolist(),
            "n": int(len(subset)),
            "median": float(subset.median()) if len(subset) else 0.0,
            "mean": round(float(subset.mean()), 1) if len(subset) else 0.0,
        })

    return {
        "kind": "histogram",
        "title": "Description length by class",
        "x_label": "Words in the description",
        "y_label": "Share of that class",
        "note": "Clipped at the 99th percentile. Densities, not counts.",
        "edges": [round(float(e), 1) for e in edges],
        "series": series,
    }


def _flags_chart(frame: pd.DataFrame) -> dict[str, Any]:
    baseline = float(frame[config.TARGET].mean() * 100)
    groups = []
    for flag in config.BINARY_FLAGS:
        rates = frame.groupby(flag)[config.TARGET].agg(["mean", "count", "sum"])
        groups.append({
            "label": flag,
            "bars": [
                {
                    "label": str(value),
                    "rate": round(float(row["mean"] * 100), 2),
                    "n": int(row["count"]),
                    "n_fraud": int(row["sum"]),
                }
                for value, row in rates.iterrows()
            ],
        })
    return {
        "kind": "grouped_bar",
        "title": "Fraud rate by binary flag",
        "y_label": "Fraud rate (%)",
        "baseline": round(baseline, 2),
        "baseline_label": f"dataset baseline {baseline:.2f}%",
        "groups": groups,
    }


def _terms_chart(
    frame: pd.DataFrame, top_n: int = 20, normalise: bool = True
) -> dict[str, Any]:
    from .explore import distinguishing_terms

    table = distinguishing_terms(
        frame, top_n=top_n, normalise_placeholders=normalise
    )
    # `distinguishing_terms` takes the top N by rank, which on a small or
    # degenerate corpus can hand back "most fraudulent" terms whose log-odds
    # are still negative. The chart labels its two halves "indicative of
    # fraudulent" and "indicative of legitimate", so a term is only allowed
    # on a side if its sign agrees with that claim.
    fraud = (
        table[(table["indicative_of"] == "fraudulent") & (table["log_odds"] > 0)]
        .nlargest(top_n, "log_odds")
    )
    legit = (
        table[(table["indicative_of"] == "legitimate") & (table["log_odds"] < 0)]
        .nsmallest(top_n, "log_odds")
    )

    def rows(part: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "term": str(row["term"]),
                "log_odds": round(float(row["log_odds"]), 3),
                "pct_fraudulent": float(row["pct_of_fraudulent"]),
                "pct_legitimate": float(row["pct_of_legitimate"]),
                "docfreq_fraudulent": int(row["docfreq_fraudulent"]),
                "docfreq_legitimate": int(row["docfreq_legitimate"]),
            }
            for _, row in part.iterrows()
        ]

    return {
        "kind": "diverging_bar",
        "title": "Distinguishing vocabulary",
        "x_label": "Smoothed log-odds ratio",
        "note": "Positive favours fraudulent, negative favours legitimate.",
        "max_items": top_n,
        "normalised": bool(normalise),
        "fraudulent": rows(fraud),
        "legitimate": rows(legit),
    }


def _missingness_chart(frame: pd.DataFrame, top_n: int = 12) -> dict[str, Any]:
    from .explore import missingness_by_class

    table = missingness_by_class(frame).head(top_n)
    return {
        "kind": "paired_bar",
        "title": "Missing fields by class",
        "x_label": "Postings where the field is empty (%)",
        "note": "Ordered by the gap between the classes.",
        "rows": [
            {
                "field": str(row["column"]),
                "legitimate": float(row["pct_missing_legitimate"]),
                "fraudulent": float(row["pct_missing_fraudulent"]),
                "difference": float(row["difference_pp"]),
            }
            for _, row in table.iterrows()
        ],
    }


def _category_chart(frame: pd.DataFrame, field: str, top_n: int = 15) -> dict[str, Any]:
    from .explore import fraud_rate_by

    if field not in config.CATEGORICAL_FIELDS:
        field = config.CATEGORICAL_FIELDS[0]
    table = fraud_rate_by(frame, field).head(top_n)
    baseline = float(frame[config.TARGET].mean() * 100)
    return {
        "kind": "ranked_bar",
        "title": f"Fraud rate by {field.replace('_', ' ')}",
        "x_label": "Fraud rate (%)",
        "field": field,
        "baseline": round(baseline, 2),
        "note": "Values with fewer than 30 postings are excluded.",
        "rows": [
            {
                "value": str(row["value"]),
                "rate": float(row["fraud_rate_pct"]),
                "n": int(row["n_postings"]),
                "n_fraud": int(row["n_fraudulent"]),
            }
            for _, row in table.iterrows()
        ],
    }


BUILDERS = {
    "length": lambda frame, _opt: _length_chart(frame),
    "flags": lambda frame, _opt: _flags_chart(frame),
    # The terms option is "normalised" / "raw" rather than a count: the
    # comparison between the two rankings is the point of the chart.
    "terms": lambda frame, opt: _terms_chart(frame, normalise=(opt != "raw")),
    "missingness": lambda frame, _opt: _missingness_chart(frame),
    "category": lambda frame, opt: _category_chart(
        frame, str(opt or config.CATEGORICAL_FIELDS[0])
    ),
}


def build(keys: list[str], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Data for each requested chart, computed once per dataset and reused."""
    from .data import load_raw

    options = options or {}
    wanted = validate_keys(keys)
    if not wanted:
        return {}

    frame = load_raw(copy=False)
    output: dict[str, Any] = {}
    for key in wanted:
        option = options.get(key)
        try:
            output[key] = _cached(
                (key, str(option)), lambda k=key, o=option: BUILDERS[k](frame, o)
            )
        except Exception:
            logger.exception("could not build chart %s", key)
            output[key] = {"error": f"Could not build the {key} chart."}
    return output
