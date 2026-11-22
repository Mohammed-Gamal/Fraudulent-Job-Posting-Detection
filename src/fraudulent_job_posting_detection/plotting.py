"""Shared output helpers: saving tables, saving figures, and the chart types
that appear in more than one stage.

Stage 1 ranks terms by log-odds and stage 5 ranks features by model
coefficient. Both draw the same picture -- two horizontal bar panels, one per
class -- so the drawing lives here once and each stage supplies its data.

Thread safety
-------------
`pyplot` keeps the "current figure" in module-level global state. The web
interface runs long actions on worker threads, and two threads drawing at the
same time interleave their calls: axes land on the wrong figure and both PNGs
come out wrong, with no error raised. `FIGURE_LOCK` serialises every drawing
sequence, and callers that draw several figures hold it across the whole
group rather than per figure.
"""

from __future__ import annotations

import logging
import threading

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

logger = logging.getLogger(__name__)

# Held by anything that touches pyplot. Re-entrant so a helper that locks can
# be called from a caller that already holds it.
FIGURE_LOCK = threading.RLock()

# The "legitimate" blue is #2F6FB5 rather than the Vega #4C78A8 it started as:
# the original sits below the chroma floor for categorical series, so against
# white it reads as a desaturated grey rather than as an identity colour. The
# replacement keeps the hue, clears the floor, and widens the colour-blind
# separation from the fraud coral (protanopia dE 12.7 -> 15.9).
PALETTE = {
    "legitimate": "#2F6FB5",
    "fraudulent": "#E45756",
    "neutral": "#6C8EBF",
    "accent": "#7D56C2",
    "grid": "#D6DBE4",
}

# Figures are shown on a light card in the interface; keep them readable there
# and in a printed report.
FIGURE_STYLE: dict[str, object] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#4A5468",
    "axes.labelcolor": "#28303F",
    "text.color": "#28303F",
    "xtick.color": "#4A5468",
    "ytick.color": "#4A5468",
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
    "savefig.facecolor": "white",
}


def save_table(frame: pd.DataFrame, name: str) -> None:
    """Write a table as CSV (for reuse) and Markdown (for the report)."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)
    (config.TABLES_DIR / f"{name}.md").write_text(
        frame.to_markdown(index=False), encoding="utf-8"
    )
    logger.info("table -> reports/tables/%s.{csv,md}", name)


def save_figure(name: str) -> None:
    """Write the current figure and close it. Caller must hold FIGURE_LOCK."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("figure -> reports/figures/%s.png", name)


def tidy(axis) -> None:
    """Drop the top and right spines and put the grid behind the data."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)


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
    with FIGURE_LOCK, plt.rc_context(FIGURE_STYLE):
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
            axis.grid(axis="y", visible=False)
            tidy(axis)
        save_figure(filename)
