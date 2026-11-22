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
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

logger = logging.getLogger(__name__)


# Normalising and factorising 17,880 descriptions takes about 0.4s, and a
# single split audit asks for it three times. The result depends only on the
# description column, so it is memoised against that column's identity.
# Bounded to a handful of entries: the keys are frames, and an unbounded map
# would pin every corpus ever read for the life of the process.
# Keyed on the identity of the frame object, not on its contents. Hashing the
# column costs 170ms, a third of the work being cached, and a sampled
# fingerprint could collide and hand back groups for the wrong data -- which
# would silently break the leakage guarantee, the one thing this module
# exists to provide. Identity cannot collide.
#
# The entry keeps a strong reference to the frame, both to bound correctness
# (a collected frame's id can be reused by a new one) and because the hot
# caller is `read_corpus(copy=False)`, which returns one shared object. Two
# entries is enough for that and caps what is pinned in memory.
_GROUP_CACHE: OrderedDict[int, tuple[pd.DataFrame, str, pd.Series]] = OrderedDict()
_GROUP_CACHE_LOCK = threading.Lock()
_GROUP_CACHE_MAX = 2


def clear_group_cache() -> None:
    with _GROUP_CACHE_LOCK:
        _GROUP_CACHE.clear()


def make_group_key(frame: pd.DataFrame, column: str = "description") -> pd.Series:
    """Assign each posting a group id; identical texts share an id.

    Postings with an empty description each get a unique id -- lumping them
    into one group would wrongly tie thousands of unrelated postings together.
    """
    key = id(frame)
    with _GROUP_CACHE_LOCK:
        entry = _GROUP_CACHE.get(key)
        if entry is not None and entry[0] is frame and entry[1] == column:
            _GROUP_CACHE.move_to_end(key)
            return entry[2].copy()

    text = (
        frame[column].fillna("").astype(str)
        .str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    groups = text.copy()
    empty = text == ""
    groups[empty] = [f"__empty_{i}__" for i in range(int(empty.sum()))]
    result = pd.Series(pd.factorize(groups)[0], index=frame.index, name="group")

    with _GROUP_CACHE_LOCK:
        _GROUP_CACHE[key] = (frame, column, result)
        while len(_GROUP_CACHE) > _GROUP_CACHE_MAX:
            _GROUP_CACHE.popitem(last=False)
    return result.copy()


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


# --------------------------------------------------------------------------
# A split the user chooses
# --------------------------------------------------------------------------
# A k-fold splitter can only produce a test set of 1/k, so it reaches 50%,
# 33%, 25%, 20%, 16.7%... and nothing in between. Where a requested fraction
# is one of those, the well-tested scikit-learn path is used -- which keeps
# the default 20% split, and therefore every number in the README, exactly as
# it was. Anything else is packed by hand.
KFOLD_TOLERANCE = 0.004
MIN_TEST_FRACTION = 0.05
MAX_TEST_FRACTION = 0.5


def _matching_n_splits(test_fraction: float) -> int | None:
    """The k for which 1/k equals `test_fraction`, if there is one."""
    for n_splits in range(2, 21):
        if abs(1.0 / n_splits - test_fraction) <= KFOLD_TOLERANCE:
            return n_splits
    return None


def grouped_stratified_split(
    frame: pd.DataFrame,
    test_fraction: float,
    seed: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """A group-disjoint, approximately stratified split at an arbitrary ratio.

    scikit-learn has `StratifiedGroupKFold` (stratified and grouped, but only
    at 1/k) and `GroupShuffleSplit` (any ratio, but not stratified). At a
    4.84% positive rate an unstratified split can leave the test set with
    wildly the wrong number of frauds, so neither alone will do.

    This packs whole groups into the test side in two passes: first the groups
    that contain fraudulent postings, largest first, until the test side holds
    its share of the frauds; then the all-legitimate groups until it holds its
    share of the rows. Taking the scarce class first is what keeps the ratio
    close -- filling by size first leaves the positives to fall where they may.

    Groups are never split, so the leakage guarantee is structural rather than
    something to verify afterwards.
    """
    labels = frame[config.TARGET].to_numpy()
    groups = make_group_key(frame).to_numpy()
    n_rows = len(frame)

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(unique_groups))
    positives = np.bincount(inverse, weights=labels, minlength=len(unique_groups))

    target_positives = labels.sum() * test_fraction
    target_rows = n_rows * test_fraction

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique_groups))

    in_test = np.zeros(len(unique_groups), dtype=bool)
    test_positives = 0.0
    test_rows = 0

    # Pass 1: the fraud-bearing groups, largest first. Ties broken by the
    # shuffle so the result varies with the seed rather than with group id.
    with_fraud = [index for index in shuffled if positives[index] > 0]
    with_fraud.sort(key=lambda index: -positives[index])
    for index in with_fraud:
        if test_positives + positives[index] <= target_positives:
            in_test[index] = True
            test_positives += positives[index]
            test_rows += sizes[index]

    # Pass 2: the all-legitimate groups, to reach the row target.
    without_fraud = [index for index in shuffled if positives[index] == 0]
    for index in without_fraud:
        if test_rows + sizes[index] <= target_rows:
            in_test[index] = True
            test_rows += sizes[index]

    test_mask = in_test[inverse]
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)

    if test_idx.size == 0 or train_idx.size == 0:
        raise ValueError(
            f"a test fraction of {test_fraction:.3f} leaves one side empty; "
            "choose something closer to the middle"
        )
    return train_idx, test_idx


def ungrouped_split(
    frame: pd.DataFrame,
    test_fraction: float,
    seed: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """A stratified split that ignores duplicate descriptions.

    Offered only so the interface can demonstrate what grouping is worth: this
    is the split most write-ups on this dataset use, and it lets identical
    adverts sit on both sides. The scores it produces are inflated, and the
    interface says so wherever it is used.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = frame[config.TARGET].to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(np.zeros(len(frame)), labels))
    return np.sort(train_idx), np.sort(test_idx)


# `StratifiedGroupKFold` takes about 2.4s over 14,658 groups, and the default
# 20% split runs through it. The interface previews a split on every slider
# release and again when training starts, so the same arrangement was being
# recomputed several times a minute. Keyed on the frame's identity and the
# three parameters that determine the result.
_SPLIT_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = (
    OrderedDict()
)
_SPLIT_CACHE_LOCK = threading.Lock()
_SPLIT_CACHE_MAX = 6


def clear_split_cache() -> None:
    with _SPLIT_CACHE_LOCK:
        _SPLIT_CACHE.clear()


def build_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
    grouped: bool = True,
    seed: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """The split the interface asks for, plus a description of how it was made.

    Returns (train_idx, test_idx, provenance). The provenance is carried into
    the results so a table of scores can never be read without knowing which
    protocol produced it -- an ungrouped split scores higher for a reason that
    has nothing to do with the model.
    """
    test_fraction = float(
        min(max(test_fraction, MIN_TEST_FRACTION), MAX_TEST_FRACTION)
    )

    n_splits = _matching_n_splits(test_fraction) if grouped else None
    if not grouped:
        method = "stratified shuffle (not grouped)"
    elif n_splits is not None and seed == config.RANDOM_STATE:
        method = f"StratifiedGroupKFold, 1 of {n_splits} folds"
    else:
        method = "grouped stratified packing"

    cache_key = (id(frame), round(test_fraction, 6), bool(grouped), int(seed))
    with _SPLIT_CACHE_LOCK:
        entry = _SPLIT_CACHE.get(cache_key)
        if entry is not None and entry[0] is frame:
            _SPLIT_CACHE.move_to_end(cache_key)
            train_idx, test_idx = entry[1], entry[2]
        else:
            entry = None

    if entry is None:
        if not grouped:
            train_idx, test_idx = ungrouped_split(frame, test_fraction, seed)
        elif n_splits is not None and seed == config.RANDOM_STATE:
            train_idx, test_idx = train_test_indices(frame, n_splits=n_splits)
        else:
            train_idx, test_idx = grouped_stratified_split(frame, test_fraction, seed)
        with _SPLIT_CACHE_LOCK:
            _SPLIT_CACHE[cache_key] = (frame, train_idx, test_idx)
            while len(_SPLIT_CACHE) > _SPLIT_CACHE_MAX:
                _SPLIT_CACHE.popitem(last=False)

    provenance: dict[str, object] = {
        "requested_test_fraction": round(test_fraction, 4),
        "actual_test_fraction": round(len(test_idx) / len(frame), 4),
        "grouped": bool(grouped),
        "seed": int(seed),
        "method": method,
    }
    return train_idx, test_idx, provenance


def cv_splitter(n_splits: int = config.CV_FOLDS) -> StratifiedGroupKFold:
    """Cross-validation splitter for model selection *within* the training set.

    Pass the matching group array as `groups=` to cross_validate/GridSearchCV,
    or the same leakage reappears inside every CV fold.
    """
    return StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
    )


def leakage_report(
    frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> dict[str, int]:
    """How much of the test set is memorisable from the training set.

    `n_memorisable_postings` is the number the interface shows: test postings
    whose description text appears verbatim somewhere in training. Under a
    grouped split it is zero by construction; under a plain stratified split
    on this dataset it is roughly a fifth of the test set, which is the entire
    argument for grouping.
    """
    groups = make_group_key(frame).to_numpy()

    # Group ids already encode "identical description", so the overlap is an
    # integer set operation rather than a pass over 17,880 strings. Doing it
    # on the text cost about 0.2s per call, and the interface calls this on
    # every slider release.
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])
    shared = np.intersect1d(train_groups, test_groups, assume_unique=True)
    memorisable = int(np.isin(groups[test_idx], shared).sum())

    # An empty description is its own group, so it can never be shared and
    # needs no special case here.
    return {
        "shared_groups": int(shared.size),
        "shared_description_texts": int(shared.size),
        "n_memorisable_postings": memorisable,
        "pct_test_memorisable": (
            round(memorisable / len(test_idx) * 100, 2) if len(test_idx) else 0.0
        ),
    }


def audit_split(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    strict: bool = True,
) -> pd.DataFrame:
    """Report the split's composition, and by default assert it does not leak.

    With `strict` the function raises on any shared group: a silent leak would
    make every downstream number meaningless, so the grouped path treats this
    as an assertion rather than decoration.

    The interface passes `strict=False` when the user has deliberately turned
    grouping off. There the overlap is the point of the exercise, so it is
    measured and displayed instead of raising.
    """
    groups = make_group_key(frame).to_numpy()
    labels = frame[config.TARGET].to_numpy()

    report = leakage_report(frame, train_idx, test_idx)
    if strict and report["shared_groups"]:
        raise AssertionError(
            f"{report['shared_groups']} description groups appear in "
            "BOTH train and test"
        )
    rows = []
    for name, idx in (("train", train_idx), ("test", test_idx)):
        subset = labels[idx]
        rows.append(
            {
                "split": name,
                "n_postings": len(idx),
                "pct_of_dataset": round(len(idx) / len(frame) * 100, 1),
                "n_fraudulent": int(subset.sum()),
                "pct_fraudulent": round(float(subset.mean() * 100), 2),
                "n_unique_groups": len(set(groups[idx])),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs.update(report)
    logger.info(
        "Split audit: %s shared groups, %s shared description strings, "
        "%s%% of test memorisable",
        report["shared_groups"],
        report["shared_description_texts"],
        report["pct_test_memorisable"],
    )
    return table
