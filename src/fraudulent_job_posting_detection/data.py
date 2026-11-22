"""Loading and integrity-checking the raw EMSCAD dataset.

The loader deliberately validates against published dataset characteristics.
A silent mismatch -- wrong file, truncated download, a Kaggle re-upload with
different columns -- would otherwise surface much later as inexplicable metrics.
"""

from __future__ import annotations

import logging
import threading

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Raised when the raw dataset is missing or does not match expectations."""


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
# The raw CSV is ~50 MB and takes a couple of seconds to parse. The web
# interface asks for it on every render, which made simply navigating the app
# re-read the whole file each time. Keyed on (size, mtime) so replacing the
# file -- which is exactly what the upload control does -- invalidates it
# without any explicit cache-busting call.
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, object] = {"key": None, "frame": None}


def _file_key() -> tuple[int, int] | None:
    try:
        stat = config.RAW_CSV.stat()
    except OSError:
        return None
    return (stat.st_size, int(stat.st_mtime_ns))


def clear_cache() -> None:
    """Forget the cached dataset. Call after writing a new raw CSV."""
    with _CACHE_LOCK:
        _CACHE["key"] = None
        _CACHE["frame"] = None


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


def load_raw(
    validate: bool = True, use_cache: bool = True, copy: bool = True
) -> pd.DataFrame:
    """Read the raw CSV and optionally assert it matches the expected schema.

    Parameters
    ----------
    validate:
        When True, raise DatasetError if the shape, columns, or target encoding
        differ from the published EMSCAD characteristics.
    use_cache:
        Serve a previously parsed copy when the file on disk has not changed.
    copy:
        Return a defensive copy, so a caller mutating the frame cannot
        corrupt what the next caller receives. Pass False only from
        read-only callers: copying 17,880 rows costs about 90ms, which is
        most of the cost of an otherwise trivial request.
    """
    if not config.RAW_CSV.exists():
        raise DatasetError(_missing_file_message())

    key = _file_key()
    if use_cache and key is not None:
        with _CACHE_LOCK:
            if _CACHE["key"] == key and _CACHE["frame"] is not None:
                cached = _CACHE["frame"]
                return cached.copy() if copy else cached  # type: ignore[union-attr]

    try:
        # EMSCAD uses empty cells for "field not supplied", which is meaningful
        # signal rather than corruption, so default NA handling is kept and the
        # missingness is turned into features later.
        frame = pd.read_csv(config.RAW_CSV, dtype={config.TARGET: "int8"})
    except (ValueError, TypeError) as exc:
        # A wrong file -- or the right file with a non-numeric target column --
        # fails in the dtype cast. Report that, rather than a pandas traceback.
        raise DatasetError(
            f"Could not parse {config.RAW_CSV.name}: {exc}\n\n"
            f"Expected the EMSCAD CSV with an integer '{config.TARGET}' column."
        ) from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(
            f"{config.RAW_CSV.name} is not readable as CSV: {exc}"
        ) from exc

    logger.info("Loaded %s rows x %s columns", f"{len(frame):,}", frame.shape[1])

    if validate:
        _validate(frame)

    if use_cache and key is not None:
        with _CACHE_LOCK:
            _CACHE["key"] = key
            _CACHE["frame"] = frame
        return frame.copy() if copy else frame
    return frame


def validate_csv(path) -> pd.DataFrame:
    """Parse and validate a candidate CSV at an arbitrary path.

    Used to check an upload *before* it replaces the working dataset. The
    alternative -- pointing the module-level path at the candidate and calling
    `load_raw` -- mutates shared state that a concurrently running background
    job may be reading, so a validation in one request could make a training
    job read the wrong file.
    """
    from pathlib import Path

    candidate = Path(path)
    if not candidate.exists():
        raise DatasetError(f"No file at {candidate}")
    try:
        frame = pd.read_csv(candidate, dtype={config.TARGET: "int8"})
    except (ValueError, TypeError) as exc:
        raise DatasetError(
            f"Could not parse {candidate.name}: {exc}\n\n"
            f"Expected the EMSCAD CSV with an integer '{config.TARGET}' column."
        ) from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(f"{candidate.name} is not readable as CSV: {exc}") from exc
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


CLASS_LABELS: dict[int, str] = {0: "legitimate (0)", 1: "fraudulent (1)"}


def class_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Absolute and relative counts of the target classes, always two rows.

    Reindexed over both classes rather than over whatever happens to be
    present. The previous version zipped a fixed two-element label list
    against `value_counts()`, so a subset or filtered frame containing one
    class raised a length-mismatch error inside what looks like a read-only
    summary -- and `balance.loc[1]` then failed for every caller downstream.
    """
    counts = (
        frame[config.TARGET]
        .value_counts()
        .reindex(list(CLASS_LABELS), fill_value=0)
        .sort_index()
    )
    total = int(counts.sum())
    percent = (counts / total * 100).round(2) if total else counts.astype(float)
    return pd.DataFrame(
        {
            "label": [CLASS_LABELS[int(value)] for value in counts.index],
            "count": counts.to_numpy(),
            "percent": percent.to_numpy(),
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
