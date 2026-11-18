"""Stage 1: exploratory data analysis.

Produces the tables and figures that justify every later modelling decision.
Each function answers one question and returns a DataFrame, so results can be
inspected in isolation and reused from a notebook.
"""

from __future__ import annotations

import logging
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .data import class_balance, load_raw, missingness  # noqa: E402

logger = logging.getLogger(__name__)

PALETTE = {"legitimate": "#4C78A8", "fraudulent": "#E45756"}

# Figures this stage produces. Declared explicitly so the interface can show
# exploratory output without sweeping up evaluation results from stage 5.
EDA_FIGURES: tuple[str, ...] = (
    "text_length_description.png",
    "fraud_rate_binary_flags.png",
    "distinguishing_terms.png",
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_table(frame: pd.DataFrame, name: str) -> None:
    """Write a table as both CSV (for reuse) and Markdown (for the report)."""
    frame.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)
    (config.TABLES_DIR / f"{name}.md").write_text(
        frame.to_markdown(index=False), encoding="utf-8"
    )
    logger.info("table -> reports/tables/%s.{csv,md}", name)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("figure -> reports/figures/%s.png", name)


# ---------------------------------------------------------------------------
# Text length
# ---------------------------------------------------------------------------
def text_length_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Word-count statistics per text field, split by class.

    Motivation: if fraudulent postings are systematically shorter, raw length
    is a cheap feature -- and also a warning that a model may key on length
    rather than content.
    """
    rows: list[dict[str, object]] = []
    for field in config.TEXT_FIELDS:
        words = frame[field].fillna("").astype(str).str.split().str.len()
        for label, mask in (
            ("legitimate", frame[config.TARGET] == 0),
            ("fraudulent", frame[config.TARGET] == 1),
        ):
            subset = words[mask]
            rows.append(
                {
                    "field": field,
                    "class": label,
                    "mean_words": round(float(subset.mean()), 1),
                    "median_words": int(subset.median()),
                    "p90_words": int(subset.quantile(0.90)),
                    "pct_empty": round(float((subset == 0).mean() * 100), 2),
                }
            )
    return pd.DataFrame(rows)


def plot_text_length(frame: pd.DataFrame) -> None:
    words = frame["description"].fillna("").astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, words.quantile(0.99), 60)
    for label, mask, colour in (
        ("legitimate", frame[config.TARGET] == 0, PALETTE["legitimate"]),
        ("fraudulent", frame[config.TARGET] == 1, PALETTE["fraudulent"]),
    ):
        ax.hist(
            words[mask], bins=bins, alpha=0.6, label=label,
            color=colour, density=True,
        )
    ax.set_xlabel("Words in description")
    ax.set_ylabel("Density")
    ax.set_title("Description length by class (densities, 99th pct clipped)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    save_figure("text_length_description")


# ---------------------------------------------------------------------------
# Fraud rate by field
# ---------------------------------------------------------------------------
def fraud_rate_by(frame: pd.DataFrame, column: str, min_count: int = 30) -> pd.DataFrame:
    """Fraud rate within each value of `column`.

    The baseline rate is ~4.84%; values far above it are candidate signals.
    Groups below `min_count` are dropped because a 2-of-3 group reads as a 67%
    fraud rate and is pure noise.
    """
    working = frame[[column, config.TARGET]].copy()
    working[column] = working[column].fillna("(missing)").astype(str)
    grouped = working.groupby(column)[config.TARGET].agg(["count", "sum"])
    grouped = grouped[grouped["count"] >= min_count]
    grouped["fraud_rate_pct"] = (grouped["sum"] / grouped["count"] * 100).round(2)
    grouped = grouped.rename(columns={"count": "n_postings", "sum": "n_fraudulent"})
    return (
        grouped.sort_values("fraud_rate_pct", ascending=False)
        .reset_index()
        .rename(columns={column: "value"})
        .assign(field=column)[
            ["field", "value", "n_postings", "n_fraudulent", "fraud_rate_pct"]
        ]
    )


def plot_binary_flags(frame: pd.DataFrame) -> None:
    baseline = frame[config.TARGET].mean() * 100
    fig, axes = plt.subplots(1, len(config.BINARY_FLAGS), figsize=(11, 3.6), sharey=True)
    for ax, flag in zip(axes, config.BINARY_FLAGS, strict=True):
        rates = frame.groupby(flag)[config.TARGET].mean() * 100
        # One colour: height already encodes the fraud rate. Colouring the two
        # bars differently would wrongly suggest they are the two classes.
        ax.bar([str(i) for i in rates.index], rates.to_numpy(), color="#6C8EBF")
        for x, value in enumerate(rates.to_numpy()):
            ax.text(x, value + 0.3, f"{value:.1f}%", ha="center", fontsize=9)
        ax.axhline(baseline, ls="--", lw=1, color="grey")
        ax.set_title(flag, fontsize=10)
        ax.set_xlabel("value")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraud rate (%)")
    fig.suptitle(
        f"Fraud rate by binary flag (dashed = {baseline:.2f}% baseline)", fontsize=11
    )
    save_figure("fraud_rate_binary_flags")


# ---------------------------------------------------------------------------
# Distinguishing vocabulary
# ---------------------------------------------------------------------------
def distinguishing_terms(
    frame: pd.DataFrame, top_n: int = 25, min_df: int = 20
) -> pd.DataFrame:
    """Terms most associated with each class by smoothed log-odds ratio.

    Uses *document* frequency per class rather than raw counts, so one posting
    repeating a word 40 times cannot dominate. Smoothing (alpha=1) keeps terms
    absent from one class finite.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    corpus = (
        frame[config.TEXT_FIELDS].fillna("").astype(str).agg(" ".join, axis=1)
    )
    vectoriser = CountVectorizer(
        min_df=min_df, stop_words="english", binary=True, max_features=40_000
    )
    matrix = vectoriser.fit_transform(corpus)
    vocabulary = np.array(vectoriser.get_feature_names_out())

    is_fraud = (frame[config.TARGET] == 1).to_numpy()
    n_fraud, n_legit = int(is_fraud.sum()), int((~is_fraud).sum())

    df_fraud = np.asarray(matrix[is_fraud].sum(axis=0)).ravel()
    df_legit = np.asarray(matrix[~is_fraud].sum(axis=0)).ravel()

    alpha = 1.0
    p_fraud = (df_fraud + alpha) / (n_fraud + 2 * alpha)
    p_legit = (df_legit + alpha) / (n_legit + 2 * alpha)
    log_odds = np.log((p_fraud / (1 - p_fraud)) / (p_legit / (1 - p_legit)))

    table = pd.DataFrame(
        {
            "term": vocabulary,
            "docfreq_fraudulent": df_fraud,
            "docfreq_legitimate": df_legit,
            "pct_of_fraudulent": (df_fraud / n_fraud * 100).round(2),
            "pct_of_legitimate": (df_legit / n_legit * 100).round(2),
            "log_odds": log_odds.round(3),
        }
    )
    top_fraud = table.nlargest(top_n, "log_odds").assign(indicative_of="fraudulent")
    top_legit = table.nsmallest(top_n, "log_odds").assign(indicative_of="legitimate")
    return pd.concat([top_fraud, top_legit], ignore_index=True)


def plot_distinguishing_terms(terms: pd.DataFrame, top_n: int = 15) -> None:
    fraud = terms[terms["indicative_of"] == "fraudulent"].nlargest(top_n, "log_odds")
    legit = terms[terms["indicative_of"] == "legitimate"].nsmallest(top_n, "log_odds")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, data, colour, title in (
        (axes[0], fraud.iloc[::-1], PALETTE["fraudulent"], "Indicative of fraud"),
        (axes[1], legit, PALETTE["legitimate"], "Indicative of legitimacy"),
    ):
        labels = [t if len(t) <= 22 else t[:19] + "..." for t in data["term"]]
        ax.barh(labels, data["log_odds"], color=colour)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("log-odds ratio")
        ax.spines[["top", "right"]].set_visible(False)
    save_figure("distinguishing_terms")


# ---------------------------------------------------------------------------
# Integrity checks that affect modelling validity
# ---------------------------------------------------------------------------
def redaction_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """How often anonymisation placeholders appear, per class.

    EMSCAD masks URLs, emails and phone numbers with tokens such as
    '#URL_<hash>#'. If their rate differs by class, a model can learn the
    *masking* instead of the fraud. Any difference here mandates stripping
    them during preprocessing.
    """
    corpus = frame[config.TEXT_FIELDS].fillna("").astype(str).agg(" ".join, axis=1)
    rows: list[dict[str, object]] = []
    for pattern in config.REDACTION_PATTERNS:
        compiled = re.compile(pattern)
        has_token = corpus.str.contains(compiled, regex=True)
        for label, mask in (
            ("legitimate", frame[config.TARGET] == 0),
            ("fraudulent", frame[config.TARGET] == 1),
        ):
            rows.append(
                {
                    "pattern": pattern,
                    "class": label,
                    "pct_postings_containing": round(
                        float(has_token[mask].mean() * 100), 2
                    ),
                }
            )
    return pd.DataFrame(rows)


def missingness_by_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing rate split by class.

    A field being absent can itself be the signal: a posting with no company
    profile is a posting with no company behind it.
    """
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if column in (config.TARGET, config.ID_COLUMN):
            continue
        legit = frame.loc[frame[config.TARGET] == 0, column].isna().mean() * 100
        fraud = frame.loc[frame[config.TARGET] == 1, column].isna().mean() * 100
        rows.append(
            {
                "column": column,
                "pct_missing_legitimate": round(float(legit), 2),
                "pct_missing_fraudulent": round(float(fraud), 2),
                "difference_pp": round(float(fraud - legit), 2),
            }
        )
    table = pd.DataFrame(rows)
    return table.reindex(
        table["difference_pp"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def duplicate_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact-duplicate description text, and whether duplicates cross classes.

    Duplicates matter for a reason students often miss: if the same posting
    appears in both the training and test split, the test score is inflated by
    memorisation. Cross-class duplicates are worse -- identical text with
    opposite labels puts a ceiling on achievable accuracy.
    """
    text = frame["description"].fillna("").astype(str).str.strip()
    non_empty = text[text != ""]
    duplicated_mask = non_empty.duplicated(keep=False)
    duplicate_text = non_empty[duplicated_mask]

    groups = frame.loc[duplicate_text.index].groupby(duplicate_text)
    label_spread = groups[config.TARGET].nunique()

    return pd.DataFrame(
        [
            {"metric": "postings with non-empty description", "value": int(len(non_empty))},
            {"metric": "postings whose description is duplicated", "value": int(duplicated_mask.sum())},
            {"metric": "distinct duplicated description texts", "value": int(groups.ngroups)},
            {"metric": "duplicate groups containing BOTH classes", "value": int((label_spread > 1).sum())},
        ]
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_eda() -> None:
    config.ensure_directories()
    frame = load_raw()

    save_table(class_balance(frame), "class_balance")
    save_table(missingness(frame), "missingness")
    save_table(missingness_by_class(frame), "missingness_by_class")
    save_table(text_length_summary(frame), "text_length_summary")

    categorical_tables = [
        fraud_rate_by(frame, column)
        for column in config.BINARY_FLAGS + config.CATEGORICAL_FIELDS
    ]
    save_table(pd.concat(categorical_tables, ignore_index=True), "fraud_rate_by_field")

    terms = distinguishing_terms(frame)
    save_table(terms, "distinguishing_terms")
    save_table(redaction_audit(frame), "redaction_audit")
    save_table(duplicate_audit(frame), "duplicate_audit")

    plot_text_length(frame)
    plot_binary_flags(frame)
    plot_distinguishing_terms(terms)

    logger.info("Stage 1 complete.")
