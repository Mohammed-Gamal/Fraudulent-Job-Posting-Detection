"""Model registry, input validation, threshold semantics, and progress.

The validation tests matter because the interface passes form input straight
into these functions. Before this, an edited request reached `REGISTRY[name]`
and produced a 500 well away from the cause.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudulent_job_posting_detection.models import (
    ALL_METRICS,
    EMBEDDING_DEFAULTS,
    IMBALANCE_STRATEGIES,
    REGISTRY,
    available_models,
    build_pipeline,
    continuous_scores,
    default_threshold,
    embedding_settings,
    operating_threshold,
    score_kind,
    validate_metric,
    validate_metrics,
    validate_model_names,
    validate_strategy,
)
from fraudulent_job_posting_detection.progress import ProgressReporter


# -------------------------------------------------------------- validation --
def test_unknown_model_names_are_dropped():
    assert validate_model_names(["LinearSVC", "DefinitelyNotAModel"]) == ["LinearSVC"]


def test_validation_returns_registry_order_not_request_order():
    names = validate_model_names(["LinearSVC", "MultinomialNB"])
    assert names == [n for n in REGISTRY if n in set(names)]


def test_validation_can_return_nothing():
    assert validate_model_names(["nope"]) == []
    assert validate_model_names([]) == []


def test_unknown_strategy_falls_back_to_class_weight():
    assert validate_strategy("'; DROP TABLE --") == "class_weight"
    for strategy in IMBALANCE_STRATEGIES:
        assert validate_strategy(strategy) == strategy


def test_unknown_metric_falls_back():
    assert validate_metric("not_a_metric") == "f1_fraud"
    assert validate_metric("roc_auc") == "roc_auc"


def test_metric_list_is_canonical_and_never_empty():
    assert validate_metrics([]) == ["f1_fraud"]
    assert validate_metrics(["bogus"]) == ["f1_fraud"]
    chosen = validate_metrics(["roc_auc", "accuracy"])
    assert chosen == [key for key in ALL_METRICS if key in {"roc_auc", "accuracy"}]


# ------------------------------------------------------- embedding settings --
def test_embedding_settings_clamp_out_of_range_values():
    settings = embedding_settings(
        embedding_dim=100_000, dropout=5.0, max_epochs=-3, sequence_length=1
    )
    assert settings["embedding_dim"] <= 256
    assert 0.0 <= settings["dropout"] <= 0.8
    assert settings["max_epochs"] >= 1
    assert settings["sequence_length"] >= 32


def test_embedding_settings_reject_junk_and_keep_defaults():
    settings = embedding_settings(embedding_dim="banana", pooling="nonsense")
    assert settings["embedding_dim"] == EMBEDDING_DEFAULTS["embedding_dim"]
    assert settings["pooling"] == EMBEDDING_DEFAULTS["pooling"]


def test_embedding_settings_ignore_unknown_keys():
    settings = embedding_settings(not_a_setting=1)
    assert "not_a_setting" not in settings


def test_embedding_settings_accept_valid_overrides():
    assert embedding_settings(pooling="mean")["pooling"] == "mean"
    assert embedding_settings(max_epochs=3)["max_epochs"] == 3


# ------------------------------------------------------------- pipelines ----
def test_smote_is_refused_for_the_embedding_model():
    """Interpolating between integer token ids produces meaningless ids."""
    if "EmbeddingNet" not in available_models():
        pytest.skip("PyTorch is not installed")
    _, applied = build_pipeline("EmbeddingNet", strategy="smote")
    assert applied == "class_weight"


def test_smote_is_kept_for_sparse_models():
    pipeline, applied = build_pipeline("LinearSVC", strategy="smote")
    assert applied == "smote"
    assert "smote" in pipeline.named_steps


def test_unknown_model_raises_in_build_pipeline():
    with pytest.raises(KeyError):
        build_pipeline("NotAModel")


def test_every_registered_model_builds():
    for name in available_models():
        pipeline, _ = build_pipeline(name)
        assert "features" in pipeline.named_steps
        assert "clf" in pipeline.named_steps


# ------------------------------------------------------------- thresholds ---
def test_default_threshold_depends_on_the_score_kind():
    assert default_threshold("margin") == 0.0
    assert default_threshold("probability") == 0.5


def test_score_kind_matches_the_estimator_interface(corpus):
    from fraudulent_job_posting_detection import config

    labels = corpus[config.TARGET].to_numpy()
    svc, _ = build_pipeline("LinearSVC", strategy="none")
    svc.fit(corpus, labels)
    assert score_kind(svc) == "margin"
    assert operating_threshold(svc) == 0.0

    forest, _ = build_pipeline("RandomForest", strategy="none")
    forest.fit(corpus, labels)
    assert score_kind(forest) == "probability"
    assert operating_threshold(forest) == 0.5


def test_continuous_scores_shape_and_finiteness(corpus):
    from fraudulent_job_posting_detection import config

    labels = corpus[config.TARGET].to_numpy()
    pipeline, _ = build_pipeline("LogisticRegression", strategy="none")
    pipeline.fit(corpus, labels)
    scores, kind = continuous_scores(pipeline, corpus)
    assert scores.shape == (len(corpus),)
    assert np.isfinite(scores).all()
    assert kind in {"margin", "probability"}


# --------------------------------------------------------------- progress ---
def test_progress_is_monotonic_across_nested_reporters():
    seen = []
    root = ProgressReporter(lambda fraction, label: seen.append(fraction))
    root(0.02, "prep")
    outer = root.span(0.88, "training")
    parts = outer.weighted({"a": 1, "b": 4})
    parts["a"](0.0)
    parts["a"].done()
    parts["b"](0.0)
    inner = parts["b"].span(0.8)
    for epoch in range(1, 4):
        inner(epoch / 3)
    parts["b"].done()
    root.span(1.0, "saving").done()
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_progress_never_goes_backwards_within_a_reporter():
    seen = []
    reporter = ProgressReporter(lambda fraction, label: seen.append(fraction))
    reporter(0.8)
    reporter(0.3)
    assert seen[-1] >= seen[0]


def test_progress_is_clamped_to_the_unit_interval():
    seen = []
    reporter = ProgressReporter(lambda fraction, label: seen.append(fraction))
    reporter(-5)
    reporter(50)
    assert all(0.0 <= value <= 1.0 for value in seen)


def test_weighted_spans_are_proportional():
    seen = []
    root = ProgressReporter(lambda fraction, label: seen.append(fraction))
    parts = root.weighted({"small": 1, "large": 9})
    parts["small"].done()
    assert seen[-1] == pytest.approx(0.1)
    parts["large"].done()
    assert seen[-1] == pytest.approx(1.0)


def test_weighted_handles_zero_total():
    root = ProgressReporter()
    parts = root.weighted({"a": 0, "b": 0})
    assert set(parts) == {"a", "b"}


def test_iterate_reports_each_item_and_completes():
    seen = []
    reporter = ProgressReporter(lambda fraction, label: seen.append((fraction, label)))
    items = list(reporter.iterate([1, 2, 3, 4], lambda index, _: f"item {index}"))
    assert items == [1, 2, 3, 4]
    assert seen[-1][0] == pytest.approx(1.0)


def test_iterate_on_empty_sequence_still_completes():
    seen = []
    reporter = ProgressReporter(lambda fraction, label: seen.append(fraction))
    assert list(reporter.iterate([])) == []
    assert seen[-1] == pytest.approx(1.0)


def test_default_reporter_is_silent():
    ProgressReporter()(0.5, "nothing should happen")


# ----------------------------------------------------- training end to end --
@pytest.mark.slow
def test_train_and_evaluate_returns_a_renderable_payload(corpus):
    from fraudulent_job_posting_detection.models import train_and_evaluate

    seen = []
    payload = train_and_evaluate(
        corpus,
        ["LinearSVC", "MultinomialNB"],
        strategy="class_weight",
        rank_by="f1_fraud",
        progress=ProgressReporter(lambda fraction, label: seen.append(fraction)),
    )
    assert {"rows", "podium", "best", "rank_by", "n_train", "n_test"} <= set(payload)
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["f1_fraud"] >= payload["rows"][1]["f1_fraud"]
    for row in payload["rows"]:
        assert {"label", "family", "strategy", "threshold", "score_kind"} <= set(row)
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


@pytest.mark.slow
def test_training_rejects_an_empty_model_selection(corpus):
    from fraudulent_job_posting_detection.models import train_and_evaluate

    with pytest.raises(ValueError, match="no usable models"):
        train_and_evaluate(corpus, ["NotAModel"])


@pytest.mark.slow
def test_saved_model_reloads_and_scores(corpus):
    from fraudulent_job_posting_detection.models import load_ui_model, train_and_evaluate

    train_and_evaluate(corpus, ["LinearSVC"], strategy="none")
    saved = load_ui_model()
    assert saved is not None
    name, pipeline, metadata = saved
    assert name == "LinearSVC"
    assert metadata["score_kind"] == "margin"
    scores, _ = continuous_scores(pipeline, corpus.head(5))
    assert np.isfinite(scores).all()
