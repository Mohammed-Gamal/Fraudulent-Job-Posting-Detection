"""Leakage-controlled train/test splitting.

Stage 1 found that 4,446 postings (24.9%) share their description text with at
least one other posting. Under an ordinary random split, identical documents
land on both sides of the boundary and the test score measures memorisation.

Every split here is *grouped by description*: all postings sharing a
description stay together, so a text seen in training can never reappear in
test. Splits remain stratified, because at a 4.84% positive rate an
unstratified split can distort the minority class badly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

logger = logging.getLogger(__name__)


def make_group_key(frame: pd.DataFrame, column: str = "description") -> pd.Series:
    """Assign each posting a group id; identical texts share an id.

    Postings with an empty description each get a unique id -- lumping them
    into one group would wrongly tie thousands of unrelated postings together.
    """
    text = (
        frame[column].fillna("").astype(str)
        .str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    groups = text.copy()
    empty = text == ""
    groups[empty] = [f"__empty_{i}__" for i in range(int(empty.sum()))]
    return pd.Series(pd.factorize(groups)[0], index=frame.index, name="group")


def train_test_indices(
    frame: pd.DataFrame, n_splits: int = 5, fold: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) positional indices for a grouped, stratified split.

    Taking one fold of a `n_splits`-way split yields a test set of roughly
    1/n_splits of the data (20% at the default of 5) that is both stratified
    and group-disjoint. `GroupShuffleSplit` would be group-aware but not
    stratified, which matters at this class ratio.
    """
    labels = frame[config.TARGET].to_numpy()
    groups = make_group_key(frame).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
    )
    for index, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(frame)), labels, groups)
    ):
        if index == fold:
            return train_idx, test_idx
    raise ValueError(f"fold {fold} out of range for n_splits={n_splits}")


def cv_splitter(n_splits: int = config.CV_FOLDS) -> StratifiedGroupKFold:
    """Cross-validation splitter for model selection *within* the training set.

    Pass the matching group array as `groups=` to cross_validate/GridSearchCV,
    or the same leakage reappears inside every CV fold.
    """
    return StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
    )


def audit_split(
    frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> pd.DataFrame:
    """Verify the split is group-disjoint and report its composition.

    This is an assertion, not decoration: a silent leak would make every
    downstream number meaningless.
    """
    groups = make_group_key(frame).to_numpy()
    labels = frame[config.TARGET].to_numpy()

    train_groups, test_groups = set(groups[train_idx]), set(groups[test_idx])
    shared = train_groups & test_groups
    if shared:
        raise AssertionError(
            f"{len(shared)} description groups appear in BOTH train and test"
        )

    text = frame["description"].fillna("").astype(str).str.strip().str.lower()
    overlap = set(text.iloc[train_idx]) & set(text.iloc[test_idx]) - {""}

    rows = []
    for name, idx in (("train", train_idx), ("test", test_idx)):
        subset = labels[idx]
        rows.append(
            {
                "split": name,
                "n_postings": len(idx),
                "n_fraudulent": int(subset.sum()),
                "pct_fraudulent": round(float(subset.mean() * 100), 2),
                "n_unique_groups": len(set(groups[idx])),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["shared_description_texts"] = len(overlap)
    logger.info(
        "Split audit: 0 shared groups, %s shared description strings", len(overlap)
    )
    return table
