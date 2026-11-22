"""The user-chosen split, and the leakage measurement that justifies it.

Two properties matter and are easy to break silently: an arbitrary ratio must
still keep whole duplicate groups on one side, and it must still land close
to the dataset's class balance. A split that quietly stopped doing either
would raise every score and look fine.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudulent_job_posting_detection import config
from fraudulent_job_posting_detection.models import DEFAULT_SPLIT, split_settings
from fraudulent_job_posting_detection.splits import (
    MAX_TEST_FRACTION,
    MIN_TEST_FRACTION,
    audit_split,
    build_split,
    grouped_stratified_split,
    leakage_report,
    make_group_key,
    train_test_indices,
    ungrouped_split,
)

RATIOS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


# ------------------------------------------------------------ the packer --
@pytest.mark.parametrize("fraction", RATIOS)
def test_packed_split_is_group_disjoint(raw_frame, fraction):
    train_idx, test_idx = grouped_stratified_split(raw_frame, fraction)
    groups = make_group_key(raw_frame).to_numpy()
    assert not (set(groups[train_idx]) & set(groups[test_idx]))


@pytest.mark.parametrize("fraction", RATIOS)
def test_packed_split_lands_near_the_requested_size(raw_frame, fraction):
    _, test_idx = grouped_stratified_split(raw_frame, fraction)
    actual = len(test_idx) / len(raw_frame)
    # Whole groups are indivisible, so an exact hit is not always possible.
    assert abs(actual - fraction) < 0.06, f"asked {fraction}, got {actual:.3f}"


@pytest.mark.parametrize("fraction", RATIOS)
def test_packed_split_preserves_the_class_ratio(raw_frame, fraction):
    train_idx, test_idx = grouped_stratified_split(raw_frame, fraction)
    labels = raw_frame[config.TARGET].to_numpy()
    overall = labels.mean()
    assert abs(labels[test_idx].mean() - overall) < 0.04
    assert abs(labels[train_idx].mean() - overall) < 0.04


def test_packed_split_covers_every_row_once(raw_frame):
    train_idx, test_idx = grouped_stratified_split(raw_frame, 0.3)
    combined = np.concatenate([train_idx, test_idx])
    assert sorted(combined.tolist()) == list(range(len(raw_frame)))


def test_packed_split_is_reproducible(raw_frame):
    first = grouped_stratified_split(raw_frame, 0.3, seed=7)
    second = grouped_stratified_split(raw_frame, 0.3, seed=7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_different_seeds_give_different_splits(raw_frame):
    _, first = grouped_stratified_split(raw_frame, 0.3, seed=1)
    _, second = grouped_stratified_split(raw_frame, 0.3, seed=99)
    assert not np.array_equal(first, second)


# ------------------------------------------------------------ build_split --
def test_the_default_still_uses_the_kfold_path(raw_frame):
    """20% must keep routing through StratifiedGroupKFold.

    Every figure quoted in the README comes from that splitter; silently
    swapping it for the packer would change them all.
    """
    train_idx, test_idx, provenance = build_split(raw_frame, 0.2)
    expected_train, expected_test = train_test_indices(raw_frame, n_splits=5)
    assert np.array_equal(test_idx, expected_test)
    assert np.array_equal(train_idx, expected_train)
    assert "StratifiedGroupKFold" in provenance["method"]


@pytest.mark.parametrize("fraction", [0.5, 0.25, 0.2, 0.1])
def test_reciprocal_fractions_use_the_kfold_path(raw_frame, fraction):
    _, _, provenance = build_split(raw_frame, fraction)
    assert "StratifiedGroupKFold" in provenance["method"]


@pytest.mark.parametrize("fraction", [0.15, 0.3, 0.35, 0.4])
def test_other_fractions_use_the_packer(raw_frame, fraction):
    _, _, provenance = build_split(raw_frame, fraction)
    assert provenance["method"] == "grouped stratified packing"


def test_a_custom_seed_leaves_the_kfold_path(raw_frame):
    """A seed the k-fold path cannot honour must not be silently ignored."""
    _, _, provenance = build_split(raw_frame, 0.2, seed=123)
    assert provenance["method"] == "grouped stratified packing"
    assert provenance["seed"] == 123


@pytest.mark.parametrize("fraction", RATIOS)
def test_grouped_split_never_leaks(raw_frame, fraction):
    train_idx, test_idx, _ = build_split(raw_frame, fraction, grouped=True)
    report = leakage_report(raw_frame, train_idx, test_idx)
    assert report["shared_groups"] == 0
    assert report["n_memorisable_postings"] == 0


def test_ungrouped_split_does_leak(raw_frame):
    """The whole argument for grouping, asserted rather than claimed."""
    train_idx, test_idx, provenance = build_split(raw_frame, 0.2, grouped=False)
    report = leakage_report(raw_frame, train_idx, test_idx)
    assert provenance["grouped"] is False
    assert report["n_memorisable_postings"] > 0
    assert report["pct_test_memorisable"] > 0


def test_out_of_range_fractions_are_clamped(raw_frame):
    _, _, low = build_split(raw_frame, 0.0001)
    _, _, high = build_split(raw_frame, 0.99)
    assert low["requested_test_fraction"] == MIN_TEST_FRACTION
    assert high["requested_test_fraction"] == MAX_TEST_FRACTION


def test_provenance_reports_the_realised_fraction(raw_frame):
    _, test_idx, provenance = build_split(raw_frame, 0.3)
    assert provenance["actual_test_fraction"] == pytest.approx(
        len(test_idx) / len(raw_frame), abs=1e-4
    )


def test_ungrouped_split_is_stratified(raw_frame):
    train_idx, test_idx = ungrouped_split(raw_frame, 0.25)
    labels = raw_frame[config.TARGET].to_numpy()
    assert abs(labels[test_idx].mean() - labels.mean()) < 0.03
    assert len(test_idx) + len(train_idx) == len(raw_frame)


# --------------------------------------------------------------- auditing --
def test_audit_reports_rather_than_raises_when_not_strict(raw_frame):
    train_idx, test_idx = ungrouped_split(raw_frame, 0.2)
    table = audit_split(raw_frame, train_idx, test_idx, strict=False)
    assert list(table["split"]) == ["train", "test"]
    assert table.attrs["n_memorisable_postings"] >= 0


def test_audit_still_raises_on_a_leak_when_strict(raw_frame):
    groups = make_group_key(raw_frame).to_numpy()
    leaked = int(np.flatnonzero(groups == groups[0])[0])
    with pytest.raises(AssertionError, match="BOTH train and test"):
        audit_split(raw_frame, np.arange(len(raw_frame)), np.array([leaked]))


def test_audit_table_shares_sum_to_one_hundred(raw_frame):
    train_idx, test_idx, _ = build_split(raw_frame, 0.25)
    table = audit_split(raw_frame, train_idx, test_idx)
    assert abs(table["pct_of_dataset"].sum() - 100.0) < 0.2


# ------------------------------------------------------------- validation --
def test_split_settings_fill_defaults():
    assert split_settings() == DEFAULT_SPLIT


def test_split_settings_clamp_and_coerce():
    settings = split_settings(test_fraction=5.0, grouped="yes", seed="abc")
    assert settings["test_fraction"] == MAX_TEST_FRACTION
    assert settings["grouped"] is True
    assert settings["seed"] == config.RANDOM_STATE


def test_split_settings_reject_junk_fractions():
    assert split_settings(test_fraction="banana")["test_fraction"] == 0.2


def test_split_settings_ignore_unknown_keys():
    assert "nonsense" not in split_settings(nonsense=1)
