"""Shared output helpers: saving tables, saving figures, and the one chart
type that appears in more than one stage.

Stage 1 ranks terms by log-odds and stage 5 ranks features by model
coefficient. Both draw the same picture -- two horizontal bar panels, one per
class -- so the drawing lives here once and each stage supplies its data.
"""

from __future__ import annotations

import logging

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

logger = logging.getLogger(__name__)

PALETTE = {"legitimate": "#4C78A8", "fraudulent": "#E45756"}


def save_table(frame: pd.DataFrame, name: str) -> None:
    """Write a table as CSV (for reuse) and Markdown (for the report)."""
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


def two_panel_barh(
    fraud_side: pd.DataFrame,
    legit_side: pd.DataFrame,
    *,
    value_column: str,
    label_column: str,
    xlabel: str,
    filename: str,
    titles: tuple[str, str] = ("Indicative of fraud", "Indicative of legitimacy"),
    max_label: int = 22,
) -> None:
    """Two horizontal bar panels, fraud on the left and legitimate on the right.

    Labels are truncated because some features are long n-grams, and matplotlib
    will happily overlap them into illegibility.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        (axes[0], fraud_side, PALETTE["fraudulent"], titles[0]),
        (axes[1], legit_side, PALETTE["legitimate"], titles[1]),
    )
    for axis, data, colour, title in panels:
        labels = [
            text if len(text) <= max_label else text[: max_label - 3] + "..."
            for text in data[label_column].astype(str)
        ]
        axis.barh(labels, data[value_column], color=colour)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(xlabel)
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(filename)
