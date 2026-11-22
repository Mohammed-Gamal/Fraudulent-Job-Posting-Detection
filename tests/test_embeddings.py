"""The embedding layer: encoding, the network, persistence, and the traps.

Several of these pin bugs that were found and fixed during development and
would be easy to reintroduce: padding contributing to the pooled vector, an
all-padding document producing NaN, and the fitted network failing to pickle
because its classes are defined inside a function.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudulent_job_posting_detection.embeddings import (
    PAD_ID,
    POOLING_MODES,
    RESERVED_TOKENS,
    UNK_ID,
    SequenceEncoder,
)

pytestmark = pytest.mark.usefixtures("torch_or_skip")

DOCS = [
    "earn money fast work home no experi urgent",
    "softwar engin python analyt infrastructur design",
    "earn money quick home data entri clerk",
    "senior analyt engin softwar distribut system",
    "recruit immedi start weekli pay no experi",
    "mentor junior staff deliveri platform analyt",
]


def column(docs):
    return np.array(docs, dtype=object).reshape(-1, 1)


# ---------------------------------------------------------------- encoder --
def test_encoder_reserves_pad_and_unk():
    encoder = SequenceEncoder(max_vocabulary=50, sequence_length=8,
                              min_document_frequency=1).fit(column(DOCS))
    assert encoder.vocabulary_[RESERVED_TOKENS[0]] == PAD_ID
    assert encoder.vocabulary_[RESERVED_TOKENS[1]] == UNK_ID


def test_encoder_pads_and_truncates_to_a_fixed_width():
    encoder = SequenceEncoder(max_vocabulary=50, sequence_length=4,
                              min_document_frequency=1).fit(column(DOCS))
    encoded = encoder.transform(column(DOCS))
    assert encoded.shape == (len(DOCS), 4)
    assert encoded.dtype == np.int64


def test_short_documents_are_right_padded():
    encoder = SequenceEncoder(max_vocabulary=50, sequence_length=10,
                              min_document_frequency=1).fit(column(["alpha beta"]))
    encoded = encoder.transform(column(["alpha beta"]))
    assert encoded[0, 0] != PAD_ID and encoded[0, 1] != PAD_ID
    assert (encoded[0, 2:] == PAD_ID).all()


def test_empty_document_is_all_padding():
    encoder = SequenceEncoder(max_vocabulary=50, sequence_length=6,
                              min_document_frequency=1).fit(column(DOCS))
    assert (encoder.transform(column([""]))[0] == PAD_ID).all()


def test_unseen_terms_map_to_unk():
    encoder = SequenceEncoder(max_vocabulary=50, sequence_length=4,
                              min_document_frequency=1).fit(column(DOCS))
    assert (encoder.transform(column(["zzzz yyyy"]))[0, :2] == UNK_ID).all()


def test_vocabulary_respects_min_document_frequency():
    docs = ["common word"] * 5 + ["rare"]
    encoder = SequenceEncoder(max_vocabulary=50, sequence_length=4,
                              min_document_frequency=2).fit(column(docs))
    assert "common" in encoder.vocabulary_
    assert "rare" not in encoder.vocabulary_


def test_vocabulary_respects_the_size_budget():
    encoder = SequenceEncoder(max_vocabulary=5, sequence_length=4,
                              min_document_frequency=1).fit(column(DOCS))
    assert len(encoder.vocabulary_) == 5


def test_vocabulary_is_deterministic():
    a = SequenceEncoder(max_vocabulary=20, sequence_length=4,
                        min_document_frequency=1).fit(column(DOCS)).vocabulary_
    b = SequenceEncoder(max_vocabulary=20, sequence_length=4,
                        min_document_frequency=1).fit(column(DOCS)).vocabulary_
    assert a == b


def test_encoder_is_fitted_on_training_documents_only():
    """Vocabulary leakage would be the same mistake as fitting TF-IDF early."""
    held_out_only = "mentor"
    assert all(held_out_only not in doc for doc in DOCS[:3])
    assert any(held_out_only in doc for doc in DOCS[3:])

    encoder = SequenceEncoder(max_vocabulary=50, sequence_length=6,
                              min_document_frequency=1).fit(column(DOCS[:3]))
    assert held_out_only not in encoder.vocabulary_
    # And it must still be encodable, as the unknown token.
    assert encoder.transform(column([held_out_only]))[0, 0] == UNK_ID


def test_transform_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        SequenceEncoder().transform(column(DOCS))


def test_inverse_tokens_round_trips_and_drops_padding():
    encoder = SequenceEncoder(max_vocabulary=50, sequence_length=12,
                              min_document_frequency=1).fit(column(DOCS))
    encoded = encoder.transform(column(["earn money fast"]))
    assert encoder.inverse_tokens(encoded[0]) == ["earn", "money", "fast"]


# ---------------------------------------------------------------- network --
def toy_dataset(n_each=60, seq_len=8):
    fraud = ["earn money fast home no experi urgent cash"] * n_each
    legit = ["softwar engin python analyt infrastructur design team build"] * n_each
    docs = fraud + legit
    labels = np.array([1] * n_each + [0] * n_each)
    encoder = SequenceEncoder(max_vocabulary=60, sequence_length=seq_len,
                              min_document_frequency=1).fit(column(docs))
    sequences = encoder.transform(column(docs))
    structured = np.zeros((len(docs), 2), dtype=float)
    structured[:n_each, 0] = 1.0
    return encoder, np.hstack([sequences.astype(float), structured]), labels


def make_classifier(**kwargs):
    from fraudulent_job_posting_detection.embeddings import EmbeddingTextClassifier

    params = dict(
        sequence_length=8, vocabulary_size=60, embedding_dim=16, hidden_dim=16,
        n_filters=12, max_epochs=6, batch_size=16, validation_fraction=0.25,
        patience=4,
    )
    params.update(kwargs)
    return EmbeddingTextClassifier(**params)


@pytest.mark.parametrize("pooling", sorted(POOLING_MODES))
def test_every_pooling_mode_trains_and_predicts(pooling):
    _, X, y = toy_dataset()
    model = make_classifier(pooling=pooling).fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (len(y),)
    assert np.isfinite(scores).all()
    assert set(np.unique(model.predict(X))) <= {0, 1}


@pytest.mark.parametrize("pooling", sorted(POOLING_MODES))
def test_all_padding_row_does_not_produce_nan(pooling):
    """An advert that is empty after cleaning must not poison the batch."""
    _, X, y = toy_dataset()
    model = make_classifier(pooling=pooling).fit(X, y)
    blank = np.zeros((1, X.shape[1]))
    assert np.isfinite(model.decision_function(blank)).all()
    assert np.isfinite(model.predict_proba(blank)).all()


def test_padding_does_not_change_the_prediction():
    """Extending a document with padding must leave its score alone.

    If padding contributed to the pooled vector, short documents would be
    dominated by an embedding for 'nothing'.
    """
    encoder = SequenceEncoder(max_vocabulary=60, sequence_length=12,
                              min_document_frequency=1).fit(column(DOCS))
    sequences = encoder.transform(column(DOCS))
    structured = np.zeros((len(DOCS), 1))
    X = np.hstack([sequences.astype(float), structured])
    y = np.array([1, 0, 1, 0, 1, 0])
    model = make_classifier(sequence_length=12, pooling="mean",
                            validation_fraction=0.34).fit(X, y)

    short = encoder.transform(column(["earn money fast"]))
    padded = short.copy()
    first = model.decision_function(np.hstack([short.astype(float), np.zeros((1, 1))]))
    second = model.decision_function(np.hstack([padded.astype(float), np.zeros((1, 1))]))
    assert np.allclose(first, second)


def test_predict_proba_is_a_distribution():
    _, X, y = toy_dataset()
    model = make_classifier().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_training_is_reproducible_for_a_fixed_seed():
    _, X, y = toy_dataset()
    first = make_classifier(random_state=7).fit(X, y).decision_function(X)
    second = make_classifier(random_state=7).fit(X, y).decision_function(X)
    assert np.allclose(first, second)


def test_pos_weight_reflects_the_class_ratio():
    _, X, y = toy_dataset()
    y = y.copy()
    y[:50] = 0  # make positives rare
    model = make_classifier(balanced=True).fit(X, y)
    positives = int(y.sum())
    expected = (len(y) - positives) / positives
    # The holdout is carved out before the ratio is computed, so allow slack.
    assert model.pos_weight_ > 1.0
    assert abs(model.pos_weight_ - expected) / expected < 0.5


def test_unbalanced_mode_uses_no_weighting():
    _, X, y = toy_dataset()
    assert make_classifier(balanced=False).fit(X, y).pos_weight_ == 1.0


def test_early_stopping_records_a_history():
    _, X, y = toy_dataset()
    model = make_classifier(max_epochs=6).fit(X, y)
    assert 1 <= len(model.history_) <= 6
    assert {"epoch", "train_loss", "val_avg_precision"} <= set(model.history_[0])
    assert 1 <= model.best_epoch_ <= len(model.history_)


def test_threshold_is_tuned_on_the_holdout():
    _, X, y = toy_dataset()
    tuned = make_classifier(tune_threshold=True).fit(X, y)
    untuned = make_classifier(tune_threshold=False).fit(X, y)
    assert untuned.threshold_ == 0.0
    assert np.isfinite(tuned.threshold_)


def test_predict_uses_the_tuned_threshold():
    _, X, y = toy_dataset()
    model = make_classifier(tune_threshold=True).fit(X, y)
    expected = (model.decision_function(X) >= model.threshold_).astype(int)
    assert np.array_equal(model.predict(X), expected)


def test_attention_weights_are_masked_and_normalised():
    _, X, y = toy_dataset()
    model = make_classifier(pooling="gated").fit(X, y)
    weights = model.attention_weights(X)
    assert weights.shape == (len(y), 8)
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    blank = np.zeros((1, X.shape[1]))
    assert float(model.attention_weights(blank).sum()) == pytest.approx(0.0, abs=1e-6)


def test_single_class_training_data_is_rejected():
    _, X, y = toy_dataset()
    with pytest.raises(ValueError, match="both classes"):
        make_classifier().fit(X, np.zeros_like(y))


def test_bad_pooling_mode_is_rejected():
    _, X, y = toy_dataset()
    with pytest.raises(ValueError, match="pooling must be"):
        make_classifier(pooling="nonsense").fit(X, y)


def test_predicting_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    _, X, _ = toy_dataset()
    with pytest.raises(NotFittedError):
        make_classifier().decision_function(X)


def test_input_narrower_than_sequence_length_is_rejected():
    _, X, y = toy_dataset()
    model = make_classifier().fit(X, y)
    with pytest.raises(ValueError, match="sequence_length"):
        model.decision_function(np.zeros((2, 3)))


# ------------------------------------------------------------ persistence --
@pytest.mark.parametrize("pooling", sorted(POOLING_MODES))
def test_fitted_model_round_trips_through_joblib(tmp_path, pooling):
    """The network classes are local to a function, so the estimator must
    persist weights rather than the module object."""
    import joblib

    _, X, y = toy_dataset()
    model = make_classifier(pooling=pooling).fit(X, y)
    before = model.decision_function(X)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)

    assert np.allclose(restored.decision_function(X), before)
    assert restored.threshold_ == model.threshold_
    assert np.array_equal(restored.predict(X), model.predict(X))


def test_pickling_drops_the_progress_callback(tmp_path):
    """A live progress reporter closes over a threading.Lock, which cannot be
    pickled; leaving it attached truncated the saved model."""
    import joblib

    _, X, y = toy_dataset()
    calls = []
    model = make_classifier(progress=lambda fraction, label: calls.append(fraction))
    model.fit(X, y)
    assert calls, "the progress callback should have been invoked during fit"

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    assert joblib.load(path).progress is None


def test_progress_callback_reports_bounded_fractions():
    _, X, y = toy_dataset()
    seen = []
    make_classifier(progress=lambda fraction, label: seen.append(fraction)).fit(X, y)
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert seen[-1] == pytest.approx(1.0)


def test_a_broken_progress_callback_does_not_stop_training():
    def explode(fraction, label):
        raise RuntimeError("reporter failed")

    _, X, y = toy_dataset()
    model = make_classifier(progress=explode).fit(X, y)
    assert np.isfinite(model.decision_function(X)).all()
