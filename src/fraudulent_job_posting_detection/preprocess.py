"""Stage 2: text normalisation.

Design notes
------------
No HTML tag stripping: an audit of the Kaggle CSV found 0% of descriptions
contain tags, though 22% carry HTML entities, so only unescaping is needed.

Redaction placeholders are *normalised, not removed*. EMSCAD masks contact
details as '#EMAIL_<sha256>#'. The hash is a per-entity identifier -- keeping
it lets a model memorise individual scammers. But the mere presence of a
contact address is genuine signal (it appears in 20.1% of fraudulent postings
against 6.8% of legitimate ones), so each family collapses to a single token.
"""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from datetime import UTC, datetime

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import config

logger = logging.getLogger(__name__)

# Canonical replacements for the three redaction families found in the data.
# Letters only, so they survive tokenisation and remain readable in feature
# importance plots.
PLACEHOLDERS: dict[str, str] = {
    "URL": "urlref",
    "EMAIL": "emailref",
    "PHONE": "phoneref",
}
PLACEHOLDER_VALUES = frozenset(PLACEHOLDERS.values())

_REDACTION_RE = re.compile(r"#([A-Za-z]+)_([0-9a-f]+)#")
_NON_LETTER_RE = re.compile(r"[^a-z]+")

_STEMMER = PorterStemmer()
_STEM_CACHE: dict[str, str] = {}

MIN_TOKEN_LENGTH = 2


def _stem(token: str) -> str:
    """Porter stem with memoisation; the corpus repeats tokens heavily."""
    cached = _STEM_CACHE.get(token)
    if cached is None:
        cached = _STEMMER.stem(token)
        _STEM_CACHE[token] = cached
    return cached


def normalise_redactions(text: str) -> str:
    """Collapse '#EMAIL_<hash>#' to 'emailref', preserving presence, dropping identity."""
    def repl(match: re.Match[str]) -> str:
        family = match.group(1).upper()
        return f" {PLACEHOLDERS.get(family, 'redactedref')} "

    return _REDACTION_RE.sub(repl, text)


def clean_text(text: str, normalise_placeholders: bool = True) -> str:
    """Full normalisation of one document: entities, redactions, case, accents, symbols."""
    if not text:
        return ""
    text = html.unescape(text)
    if normalise_placeholders:
        text = normalise_redactions(text)
    text = text.lower()
    # Fold accented characters to ASCII; 78% of descriptions contain non-ASCII
    # characters, mostly typographic quotes, dashes and bullets.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return _NON_LETTER_RE.sub(" ", text).strip()


def tokenise(text: str, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    """Split cleaned text into tokens, optionally filtering and stemming.

    Placeholder tokens bypass stopword removal and stemming so they stay
    intact and identifiable.
    """
    tokens: list[str] = []
    for token in text.split():
        if token in PLACEHOLDER_VALUES:
            tokens.append(token)
            continue
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if remove_stopwords and token in ENGLISH_STOP_WORDS:
            continue
        tokens.append(_stem(token) if stem else token)
    return tokens


def preprocess_document(
    text: str,
    remove_stopwords: bool = True,
    stem: bool = True,
    normalise_placeholders: bool = True,
) -> str:
    """Clean and tokenize one document; the three switches are exposed in the GUI."""
    cleaned = clean_text(text, normalise_placeholders=normalise_placeholders)
    return " ".join(tokenise(cleaned, remove_stopwords=remove_stopwords, stem=stem))


CHUNK_SIZE = 1_000


def build_corpus(
    frame: pd.DataFrame, progress: object = None, **options: bool
) -> pd.DataFrame:
    """Add cleaned text columns plus structural features to a copy of `frame`.

    Returns a new frame with:
      - 'text_all'      every text field concatenated, cleaned and stemmed
      - missing-value indicators, which Stage 1 showed are predictive
        (company_profile is absent in 67.8% of fraudulent postings)

    Cleaning is the slowest step in the project and the one a user is most
    likely to be watching, so it runs in chunks and reports after each. That
    is the only reason for the chunking: the per-document work is identical.
    """
    from .progress import ProgressReporter

    reporter = progress if progress is not None else ProgressReporter()
    result = frame.copy()
    n_rows = len(result)

    # Clean each field, concatenate into one document, then discard the
    # per-field copies: storing all five again would quintuple the file for
    # no modelling benefit. Per-field experiments can call build_corpus again.
    cleaned: dict[str, pd.Series] = {}
    fields = config.TEXT_FIELDS
    per_field = reporter.weighted({field: 1.0 for field in fields})

    for field in fields:
        source = result[field].fillna("").astype(str)
        step = per_field[field]
        pieces: list[pd.Series] = []
        n_chunks = max(1, -(-n_rows // CHUNK_SIZE))
        for index in range(n_chunks):
            start = index * CHUNK_SIZE
            chunk = source.iloc[start : start + CHUNK_SIZE]
            pieces.append(
                chunk.map(lambda text: preprocess_document(text, **options))
            )
            step(
                (index + 1) / n_chunks,
                f"cleaning {field} ({min(start + CHUNK_SIZE, n_rows):,}/{n_rows:,})",
            )
        cleaned[field] = (
            pd.concat(pieces) if len(pieces) > 1 else pieces[0]
        ) if pieces else source
        result[f"{field}_is_missing"] = frame[field].isna().astype("int8")

    result["text_all"] = (
        pd.DataFrame(cleaned)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    for field in config.CATEGORICAL_FIELDS:
        result[f"{field}_is_missing"] = frame[field].isna().astype("int8")

    result["n_tokens"] = result["text_all"].str.split().str.len()
    logger.info(
        "Built corpus: median %s tokens per posting, vocabulary cache %s stems",
        int(result["n_tokens"].median()) if n_rows else 0, f"{len(_STEM_CACHE):,}",
    )
    return result


DEFAULT_OPTIONS: dict[str, bool] = {
    "remove_stopwords": True,
    "stem": True,
    "normalise_placeholders": True,
}

CORPUS_PATH = config.PROCESSED_DIR / "corpus.parquet"
CORPUS_META_PATH = config.PROCESSED_DIR / "corpus_meta.json"


def read_corpus_meta() -> dict[str, object] | None:
    """How the corpus on disk was actually built, or None if unknown.

    Without this the interface cannot tell which options produced the stored
    corpus: a user could disable stemming, rebuild, restart, and then be shown
    the default settings while training on a corpus built differently. Any
    reported methodology would be wrong, and nothing would look broken.
    """
    if not CORPUS_META_PATH.exists():
        return None
    try:
        return json.loads(CORPUS_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_preprocess(progress: object = None, **options: bool) -> pd.DataFrame:
    """Build and persist the cleaned corpus, recording how it was built."""
    from .data import load_raw
    from .progress import ProgressReporter

    reporter = progress if progress is not None else ProgressReporter()
    config.ensure_directories()
    settings = {key: bool(value) for key, value in {**DEFAULT_OPTIONS, **options}.items()
                if key in DEFAULT_OPTIONS}

    stages = reporter.weighted({"load": 1.0, "clean": 8.0, "write": 1.0})
    stages["load"](0.1, "reading the raw dataset")
    frame = load_raw()
    stages["load"].done(f"loaded {len(frame):,} postings")

    processed = build_corpus(frame, progress=stages["clean"], **settings)

    stages["write"](0.2, "writing corpus.parquet")
    # Write to a temporary file and move into place: a crash mid-write would
    # otherwise leave a truncated parquet that every later step fails to read.
    temporary = CORPUS_PATH.with_suffix(".parquet.tmp")
    processed.to_parquet(temporary, index=False)
    temporary.replace(CORPUS_PATH)

    CORPUS_META_PATH.write_text(
        json.dumps(
            {
                "options": settings,
                "built_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                "rows": int(len(processed)),
                "median_tokens": int(processed["n_tokens"].median()) if len(processed) else 0,
                "vocabulary_stems": len(_STEM_CACHE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stages["write"].done("corpus written")
    reporter.done("corpus ready")
    logger.info("corpus -> %s (options: %s)", CORPUS_PATH.name, settings)
    return processed


def build_single_posting(values: dict[str, object], **options: bool) -> pd.DataFrame:
    """Turn one pasted job advert into a single-row frame the models accept.

    Routed through `build_corpus`, the same function used for the training
    data, so a posting typed into the interface is cleaned identically to the
    corpus. Doing this by hand is how train/serve skew creeps in.
    """
    row: dict[str, object] = {}
    for field in config.TEXT_FIELDS:
        text = str(values.get(field, "") or "").strip()
        row[field] = text or None
    for field in config.CATEGORICAL_FIELDS:
        row[field] = values.get(field) or None
    for flag in config.BINARY_FLAGS:
        row[flag] = int(bool(values.get(flag, 0)))

    frame = pd.DataFrame([row])
    settings = {**DEFAULT_OPTIONS, **options}
    return build_corpus(frame, **settings)
