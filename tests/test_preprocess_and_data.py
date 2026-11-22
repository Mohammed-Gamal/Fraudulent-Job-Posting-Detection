"""Loading, validation and text normalisation.

The preprocessing tests pin the behaviour the README makes claims about:
redaction placeholders are normalised rather than stripped, and a posting
typed into the interface is cleaned by exactly the same code as the corpus.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fraudulent_job_posting_detection import config
from fraudulent_job_posting_detection.data import (
    DatasetError,
    class_balance,
    clear_cache,
    load_raw,
    missingness,
    validate_csv,
)
from fraudulent_job_posting_detection.preprocess import (
    build_single_posting,
    clean_text,
    normalise_redactions,
    preprocess_document,
    read_corpus_meta,
    run_preprocess,
    tokenise,
)


# ----------------------------------------------------------------- loading --
def test_load_raw_reads_the_fixture(project):
    frame = load_raw()
    assert len(frame) > 0
    assert config.TARGET in frame.columns


def test_load_raw_errors_clearly_when_absent(project):
    project["RAW_CSV"].unlink()
    clear_cache()
    with pytest.raises(DatasetError, match="Raw dataset not found"):
        load_raw()


def test_validation_rejects_an_unrelated_csv(project, tmp_path):
    other = tmp_path / "wrong.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(other, index=False)
    with pytest.raises(DatasetError):
        validate_csv(other)


def test_validation_rejects_duplicate_ids(project, tmp_path, raw_frame):
    broken = raw_frame.copy()
    broken.loc[broken.index[1], config.ID_COLUMN] = broken.loc[broken.index[0], config.ID_COLUMN]
    path = tmp_path / "dupes.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(DatasetError, match="duplicate"):
        validate_csv(path)


def test_cache_is_invalidated_when_the_file_changes(project, raw_frame):
    first = load_raw()
    shorter = raw_frame.iloc[:50]
    # Row count only warns, so this still loads; the point is the new content.
    shorter.to_csv(project["RAW_CSV"], index=False)
    second = load_raw()
    assert len(first) != len(second)


def test_cache_returns_independent_copies(project):
    first = load_raw()
    first.loc[first.index[0], "title"] = "MUTATED"
    second = load_raw()
    assert second.loc[second.index[0], "title"] != "MUTATED"


# ----------------------------------------------------------- class balance --
def test_class_balance_always_reports_both_classes(raw_frame):
    """A frame with one class must not raise or return a short table.

    The original implementation zipped a fixed two-label list against
    `value_counts()`, so any single-class subset raised a length mismatch.
    """
    only_legit = raw_frame[raw_frame[config.TARGET] == 0]
    table = class_balance(only_legit)
    assert len(table) == 2
    assert float(table.loc[1, "percent"]) == 0.0
    assert int(table.loc[1, "count"]) == 0


def test_class_balance_on_an_empty_frame(raw_frame):
    table = class_balance(raw_frame.iloc[0:0])
    assert len(table) == 2
    assert table["count"].sum() == 0


def test_class_balance_percentages_sum_to_100(raw_frame):
    table = class_balance(raw_frame)
    assert abs(table["percent"].sum() - 100.0) < 0.05


def test_missingness_lists_every_column(raw_frame):
    table = missingness(raw_frame)
    assert set(table["column"]) == set(raw_frame.columns)


# ------------------------------------------------------------- normalising --
def test_redaction_hashes_collapse_to_one_token():
    """Keeping the hash lets the model memorise an individual scammer."""
    text = "contact #EMAIL_abc123def# or #EMAIL_999fff# today"
    out = normalise_redactions(text)
    assert "abc123def" not in out and "999fff" not in out
    assert out.count("emailref") == 2


def test_each_redaction_family_has_its_own_token():
    out = normalise_redactions("#URL_aa# #EMAIL_bb# #PHONE_cc#")
    assert "urlref" in out and "emailref" in out and "phoneref" in out


def test_unknown_redaction_family_still_normalises():
    assert "redactedref" in normalise_redactions("#FAXNUM_abc123#")


def test_clean_text_unescapes_entities_and_folds_accents():
    assert "&amp;" not in clean_text("R&amp;D")
    assert clean_text("café naïve") == "cafe naive"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert preprocess_document("") == ""


def test_placeholders_survive_tokenisation_intact():
    """They must not be stemmed or dropped as stopwords, or the signal is lost."""
    tokens = tokenise("emailref urlref phoneref the and", remove_stopwords=True, stem=True)
    assert {"emailref", "urlref", "phoneref"} <= set(tokens)


def test_stopwords_and_short_tokens_are_dropped():
    tokens = tokenise("the a of recruiting", remove_stopwords=True, stem=True)
    assert "the" not in tokens and "of" not in tokens
    assert "recruit" in tokens


def test_stemming_conflates_a_word_family():
    stems = {preprocess_document(word) for word in ("recruiting", "recruiter", "recruitment")}
    assert len(stems) < 3


def test_stemming_can_be_switched_off():
    assert preprocess_document("recruiting", stem=False) == "recruiting"


# ----------------------------------------------------------------- corpus ---
def test_run_preprocess_writes_corpus_and_provenance(project):
    frame = run_preprocess()
    from fraudulent_job_posting_detection.preprocess import CORPUS_PATH

    assert CORPUS_PATH.exists()
    meta = read_corpus_meta()
    assert meta is not None
    assert meta["rows"] == len(frame)
    assert set(meta["options"]) == {"remove_stopwords", "stem", "normalise_placeholders"}


def test_provenance_records_the_options_actually_used(project):
    run_preprocess(stem=False)
    meta = read_corpus_meta()
    assert meta["options"]["stem"] is False


def test_corpus_has_the_columns_the_models_need(corpus):
    from fraudulent_job_posting_detection.models import (
        STRUCTURED_BINARY,
        STRUCTURED_NUMERIC,
        TEXT_COLUMN,
    )

    for column in [TEXT_COLUMN, *STRUCTURED_BINARY, *STRUCTURED_NUMERIC]:
        assert column in corpus.columns, column


def test_missingness_indicators_are_binary(corpus):
    for column in corpus.columns:
        if column.endswith("_is_missing"):
            assert set(corpus[column].unique()) <= {0, 1}


def test_single_posting_matches_the_corpus_schema(corpus):
    """Train/serve skew check: a pasted advert must look like a corpus row."""
    from fraudulent_job_posting_detection.models import (
        STRUCTURED_BINARY,
        STRUCTURED_NUMERIC,
        TEXT_COLUMN,
    )

    row = build_single_posting({
        "title": "Data Entry Clerk",
        "description": "Earn money fast working from home",
        "has_company_logo": 0,
    })
    assert len(row) == 1
    for column in [TEXT_COLUMN, *STRUCTURED_BINARY, *STRUCTURED_NUMERIC]:
        assert column in row.columns, column


def test_single_posting_is_cleaned_identically_to_the_corpus():
    """The same text through both paths must produce the same tokens."""
    text = "Earn MONEY fast — recruiting now! Contact #EMAIL_deadbeef# today."
    direct = preprocess_document(text)
    through_frame = build_single_posting({"description": text})
    assert through_frame.loc[0, "text_all"].strip() == direct.strip()


def test_single_posting_marks_absent_fields_as_missing():
    row = build_single_posting({"description": "something"})
    assert int(row.loc[0, "company_profile_is_missing"]) == 1
    assert int(row.loc[0, "description_is_missing"]) == 0
