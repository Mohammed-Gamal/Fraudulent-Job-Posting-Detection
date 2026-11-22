"""The leakage control is the project's central methodological claim.

If grouping silently stopped working, every reported number would rise and
nothing would look broken. These tests fail loudly instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudulent_job_posting_detection import config
from fraudulent_job_posting_detection.splits import (
    audit_split,
    cv_splitter,
    make_group_key,
    train_test_indices,
)


def test_identical_descriptions_share_a_group(raw_frame):
    groups = make_group_key(raw_frame)
    duplicated = raw_frame[
        raw_frame["description"].str.contains("shared duplicated", na=False)
    ]
    assert len(duplicated) > 1
    assert groups.loc[duplicated.index].nunique() == 1


def test_empty_descriptions_do_not_collapse_into_one_group(raw_frame):
    """Blank text is 'no description', not 'the same description'.

    Lumping every empty posting together would tie thousands of unrelated
    adverts into a single group and distort the split.
    """
    frame = raw_frame.copy()
    frame.loc[frame.index[:6], "description"] = ""
    groups = make_group_key(frame)
    assert groups.iloc[:6].nunique() == 6


def test_group_key_ignores_case_and_whitespace(raw_frame):
    frame = raw_frame.copy()
    frame.loc[frame.index[0], "description"] = "  A Shared   Description  "
    frame.loc[frame.index[1], "description"] = "a shared description"
    groups = make_group_key(frame)
    assert groups.iloc[0] == groups.iloc[1]


def test_train_test_split_is_group_disjoint(raw_frame):
    train_idx, test_idx = train_test_indices(raw_frame)
    groups = make_group_key(raw_frame).to_numpy()
    assert not (set(groups[train_idx]) & set(groups[test_idx]))


def test_no_description_text_appears_on_both_sides(raw_frame):
    """The property that actually matters, checked on the text itself."""
    train_idx, test_idx = train_test_indices(raw_frame)
    text = raw_frame["description"].fillna("").str.strip().str.lower()
    shared = set(text.iloc[train_idx]) & set(text.iloc[test_idx])
    assert shared - {""} == set()


def test_split_covers_every_row_exactly_once(raw_frame):
    train_idx, test_idx = train_test_indices(raw_frame)
    combined = np.concatenate([train_idx, test_idx])
    assert len(combined) == len(set(combined.tolist())) == len(raw_frame)


def test_split_keeps_both_classes_on_both_sides(raw_frame):
    train_idx, test_idx = train_test_indices(raw_frame)
    labels = raw_frame[config.TARGET].to_numpy()
    assert set(labels[train_idx]) == {0, 1}
    assert set(labels[test_idx]) == {0, 1}


def test_split_is_reproducible(raw_frame):
    first = train_test_indices(raw_frame)
    second = train_test_indices(raw_frame)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_audit_split_accepts_a_clean_split(raw_frame):
    train_idx, test_idx = train_test_indices(raw_frame)
    table = audit_split(raw_frame, train_idx, test_idx)
    assert list(table["split"]) == ["train", "test"]
    assert table["n_postings"].sum() == len(raw_frame)


def test_audit_split_rejects_a_leaking_split(raw_frame):
    """A deliberately contaminated split must be caught, not reported."""
    groups = make_group_key(raw_frame).to_numpy()
    leaked = int(np.flatnonzero(groups == groups[0])[0])
    train_idx = np.arange(len(raw_frame))
    test_idx = np.array([leaked])
    with pytest.raises(AssertionError, match="BOTH train and test"):
        audit_split(raw_frame, train_idx, test_idx)


def test_cv_splitter_folds_are_group_disjoint(raw_frame):
    groups = make_group_key(raw_frame).to_numpy()
    labels = raw_frame[config.TARGET].to_numpy()
    splitter = cv_splitter(n_splits=3)
    for train_idx, test_idx in splitter.split(
        np.zeros(len(raw_frame)), labels, groups
    ):
        assert not (set(groups[train_idx]) & set(groups[test_idx]))


def test_fold_out_of_range_raises(raw_frame):
    with pytest.raises(ValueError, match="out of range"):
        train_test_indices(raw_frame, n_splits=3, fold=9)
