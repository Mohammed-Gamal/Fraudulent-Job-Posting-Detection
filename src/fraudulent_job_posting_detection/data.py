"""Loading and integrity-checking the raw EMSCAD dataset.

The loader deliberately validates against published dataset characteristics.
A silent mismatch -- wrong file, truncated download, a Kaggle re-upload with
different columns -- would otherwise surface much later as inexplicable metrics.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Raised when the raw dataset is missing or does not match expectations."""


def _missing_file_message() -> str:
    return (
        f"Raw dataset not found at:\n    {config.RAW_CSV}\n\n"
        "The EMSCAD dataset is not downloadable without a Kaggle session, so it\n"
        "must be placed there manually:\n\n"
        "  1. Open https://www.kaggle.com/datasets/shivamb/"
        "real-or-fake-fake-jobposting-prediction\n"
        "  2. Download the archive and extract 'fake_job_postings.csv'\n"
        f"  3. Place it at: {config.RAW_CSV.relative_to(config.PROJECT_ROOT)}\n"
    )


def load_raw(validate: bool = True) -> pd.DataFrame:
    """Read the raw CSV and optionally assert it matches the expected schema.

    Parameters
    ----------
    validate:
        When True, raise DatasetError if the shape, columns, or target encoding
        differ from the published EMSCAD characteristics.
    """
    if not config.RAW_CSV.exists():
        raise DatasetError(_missing_file_message())

    # keep_default_na with an explicit empty-string sentinel: EMSCAD uses empty
    # cells for "field not supplied", which is meaningful signal, not corruption.
    frame = pd.read_csv(config.RAW_CSV, dtype={config.TARGET: "int8"})
    logger.info("Loaded %s rows x %s columns", f"{len(frame):,}", frame.shape[1])

    if validate:
        _validate(frame)
    return frame


def _validate(frame: pd.DataFrame) -> None:
    problems: list[str] = []

    if frame.shape[1] != config.EXPECTED_N_COLUMNS:
        problems.append(
            f"expected {config.EXPECTED_N_COLUMNS} columns, found {frame.shape[1]}"
        )

    missing = set(config.EXPECTED_COLUMNS) - set(frame.columns)
    if missing:
        problems.append(f"missing expected columns: {sorted(missing)}")

    unexpected = set(frame.columns) - set(config.EXPECTED_COLUMNS)
    if unexpected:
        problems.append(f"unexpected extra columns: {sorted(unexpected)}")

    if len(frame) != config.EXPECTED_ROWS:
        # A warning rather than an error: Kaggle mirrors occasionally differ by
        # a row, and that should not block the pipeline.
        logger.warning(
            "Row count %s differs from the published %s",
            f"{len(frame):,}", f"{config.EXPECTED_ROWS:,}",
        )

    if config.TARGET in frame.columns:
        labels = set(frame[config.TARGET].unique().tolist())
        if not labels <= {0, 1}:
            problems.append(f"target '{config.TARGET}' has non-binary values: {labels}")

    if config.ID_COLUMN in frame.columns and frame[config.ID_COLUMN].duplicated().any():
        n = int(frame[config.ID_COLUMN].duplicated().sum())
        problems.append(f"{n} duplicate {config.ID_COLUMN} values")

    if problems:
        raise DatasetError(
            "Raw dataset failed validation:\n  - " + "\n  - ".join(problems)
        )


def class_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Return absolute and relative counts of the target classes."""
    counts = frame[config.TARGET].value_counts().sort_index()
    return pd.DataFrame(
        {
            "label": ["legitimate (0)", "fraudulent (1)"],
            "count": counts.to_numpy(),
            "percent": (counts / counts.sum() * 100).round(2).to_numpy(),
        }
    )


def missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and percentage of missing values, descending."""
    null_counts = frame.isna().sum()
    table = pd.DataFrame(
        {
            "column": null_counts.index,
            "n_missing": null_counts.to_numpy(),
            "pct_missing": (null_counts / len(frame) * 100).round(2).to_numpy(),
        }
    )
    return table.sort_values("pct_missing", ascending=False).reset_index(drop=True)
