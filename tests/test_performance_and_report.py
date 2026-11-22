"""Caching, the slimmed state endpoint, and the report stages.

The caches here exist to keep the page responsive, but a cache that returns
the wrong thing is far worse than a slow one -- the group-key cache in
particular feeds the leakage guarantee. These tests pin correctness first and
the laziness second.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fraudulent_job_posting_detection import analysis, config
from fraudulent_job_posting_detection.data import clear_cache, load_raw
from fraudulent_job_posting_detection.models import (
    clear_corpus_cache,
    read_corpus,
    saved_model_summary,
)
from fraudulent_job_posting_detection.splits import (
    build_split,
    clear_group_cache,
    clear_split_cache,
    leakage_report,
    make_group_key,
)


# ----------------------------------------------------------------- caches --
def test_corpus_cache_returns_equal_frames(corpus, project):
    clear_corpus_cache()
    first = read_corpus()
    second = read_corpus()
    assert first is not second          # copies by default
    assert first.equals(second)
    assert read_corpus(copy=False) is read_corpus(copy=False)


def test_corpus_cache_is_invalidated_by_a_rebuild(project):
    from fraudulent_job_posting_detection.preprocess import run_preprocess

    run_preprocess()
    first = read_corpus(copy=False)
    # A rebuild with different options writes a different corpus.
    time.sleep(0.01)
    run_preprocess(stem=False)
    second = read_corpus(copy=False)
    assert first is not second
    assert not first["text_all"].equals(second["text_all"])


def test_mutating_a_copy_cannot_corrupt_the_cache(corpus, project):
    frame = read_corpus()
    frame.loc[frame.index[0], "text_all"] = "MUTATED"
    assert read_corpus().loc[0, "text_all"] != "MUTATED"


def test_group_key_cache_returns_the_same_groups(corpus):
    clear_group_cache()
    first = make_group_key(corpus)
    second = make_group_key(corpus)
    assert first.equals(second)
    assert first is not second          # callers get their own copy


def test_group_key_is_not_shared_between_different_frames(raw_frame):
    """Identity keying must never hand one frame's groups to another."""
    clear_group_cache()
    other = raw_frame.copy()
    other["description"] = "every posting now has the same description"
    assert make_group_key(raw_frame).nunique() > 1
    assert make_group_key(other).nunique() == 1
    # And back again, to catch a cache that overwrote the first entry wrongly.
    assert make_group_key(raw_frame).nunique() > 1


def test_mutating_a_returned_group_key_does_not_poison_the_cache(corpus):
    clear_group_cache()
    first = make_group_key(corpus)
    first.iloc[0] = -999
    assert make_group_key(corpus).iloc[0] != -999


def test_split_cache_returns_identical_indices(corpus):
    clear_split_cache()
    train_a, test_a, _ = build_split(corpus, 0.25)
    train_b, test_b, _ = build_split(corpus, 0.25)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)


def test_split_cache_distinguishes_its_parameters(corpus):
    clear_split_cache()
    _, test_default, _ = build_split(corpus, 0.25, grouped=True, seed=42)
    _, test_ungrouped, _ = build_split(corpus, 0.25, grouped=False, seed=42)
    _, test_other_seed, _ = build_split(corpus, 0.25, grouped=True, seed=7)
    _, test_other_size, _ = build_split(corpus, 0.4, grouped=True, seed=42)
    sizes = {len(test_default), len(test_other_size)}
    assert len(sizes) == 2
    assert not np.array_equal(test_default, test_ungrouped)
    assert not np.array_equal(test_default, test_other_seed)


def test_leakage_report_agrees_with_a_text_comparison(raw_frame):
    """The fast group-id version must match the obvious string version."""
    for grouped in (True, False):
        train_idx, test_idx, _ = build_split(raw_frame, 0.25, grouped=grouped)
        text = raw_frame["description"].fillna("").astype(str).str.strip().str.lower()
        train_text = set(text.iloc[train_idx]) - {""}
        expected = int(text.iloc[test_idx].isin(train_text).sum())
        assert leakage_report(raw_frame, train_idx, test_idx)[
            "n_memorisable_postings"
        ] == expected


# ------------------------------------------------------- the state endpoint --
def test_state_omits_the_expensive_panel_payloads(client):
    """These used to be computed on every page load, for panels most
    visits never open."""
    state = client.get("/api/state").get_json()
    for key in ("split", "explore", "corpus_sample", "eda_figures"):
        assert key not in state, f"{key} should be fetched by its own panel"


def test_state_carries_what_the_shell_needs(client):
    state = client.get("/api/state").get_json()
    for key in (
        "dataset", "corpus_ready", "model", "jobs", "active_job",
        "split_defaults", "analysis", "analysis_figures",
    ):
        assert key in state


def test_state_does_not_import_torch(client):
    """Deciding whether to offer the embedding model must not cost 2.5s."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "from fraudulent_job_posting_detection.webapp.app import create_app;"
        "a=create_app(); a.config['TESTING']=True;"
        "a.test_client().get('/');"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.stdout.strip().endswith("False"), result.stdout + result.stderr


def test_explore_tables_endpoint(client):
    payload = client.get("/api/explore/tables").get_json()
    assert payload["ok"] is True
    assert payload["preview"]["rows"]
    assert "Missing values by column" in payload["tables"]


def test_corpus_sample_endpoint_needs_a_corpus(client):
    assert client.get("/api/corpus/sample").status_code == 409


def test_corpus_sample_endpoint(client):
    from fraudulent_job_posting_detection.preprocess import run_preprocess

    run_preprocess()
    payload = client.get("/api/corpus/sample").get_json()
    assert payload["ok"] is True
    assert payload["sample"]
    assert {"label", "text", "n_tokens"} <= set(payload["sample"][0])


def test_saved_model_summary_without_a_model(project):
    assert saved_model_summary() is None


@pytest.mark.slow
def test_saved_model_summary_avoids_unpickling(corpus, project):
    from fraudulent_job_posting_detection.models import (
        UI_MODEL_META_PATH,
        train_and_evaluate,
    )

    train_and_evaluate(corpus, ["LinearSVC"], strategy="none")
    assert UI_MODEL_META_PATH.exists()
    summary = saved_model_summary()
    assert summary["name"] == "LinearSVC"
    assert summary["metadata"]["score_kind"] == "margin"


@pytest.mark.slow
def test_model_summary_backfills_a_missing_sidecar(corpus, project):
    from fraudulent_job_posting_detection.models import (
        UI_MODEL_META_PATH,
        train_and_evaluate,
    )

    train_and_evaluate(corpus, ["LinearSVC"], strategy="none")
    UI_MODEL_META_PATH.unlink()
    assert saved_model_summary()["name"] == "LinearSVC"
    assert UI_MODEL_META_PATH.exists(), "the sidecar should be rewritten"


# ----------------------------------------------------------- report stages --
def test_every_report_stage_is_registered():
    assert set(analysis.STAGE_KEYS) == set(analysis.RUNNERS)
    for spec in analysis.catalogue():
        assert {"key", "label", "blurb", "cost", "controls"} <= set(spec)


def test_report_stages_are_runnable_endpoints(client):
    for key in analysis.STAGE_KEYS:
        response = client.post(f"/api/run/{key}", json={})
        # Either it starts, or it refuses for a stated reason — never 404.
        assert response.status_code in {200, 400, 409}, key
        if response.status_code == 200:
            from tests.test_webapp import wait_for_job

            wait_for_job(client, key)


def test_report_stages_need_a_corpus(client):
    for key in ("report_compare", "report_imbalance", "report_evaluate"):
        response = client.post(f"/api/run/{key}", json={})
        assert response.status_code == 400
        assert "corpus" in response.get_json()["error"].lower()


def test_analysis_results_round_trip(project):
    analysis._store("report_eda", {"ran_at": "now", "tables": []})
    assert analysis.load_analysis()["report_eda"]["ran_at"] == "now"
    analysis._store("report_imbalance", {"ran_at": "later"})
    both = analysis.load_analysis()
    assert set(both) == {"report_eda", "report_imbalance"}


def test_analysis_file_is_tolerant_of_corruption(project):
    analysis.ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    analysis.ANALYSIS_PATH.write_text("not json", encoding="utf-8")
    assert analysis.load_analysis() == {}


@pytest.mark.slow
def test_report_eda_writes_tables_and_figures(client, project):
    from tests.test_webapp import wait_for_job

    assert client.post("/api/run/report_eda", json={}).status_code == 200
    job = wait_for_job(client, "report_eda")
    assert job["status"] == "done", job["error"]
    stored = client.get("/api/state").get_json()["analysis"]["report_eda"]
    assert stored["tables"]
    assert (config.TABLES_DIR / "class_balance.csv").exists()
    assert (config.TABLES_DIR / "class_balance.md").exists()


@pytest.mark.slow
def test_report_compare_is_resumable(client, corpus, project):
    from tests.test_webapp import wait_for_job

    body = {"models": ["MultinomialNB"], "folds": 2, "feature_sets": ["tfidf only"]}
    assert client.post("/api/run/report_compare", json=body).status_code == 200
    assert wait_for_job(client, "report_compare")["status"] == "done"
    first = client.get("/api/state").get_json()["analysis"]["report_compare"]

    # A second run with the same selection should reuse what is on disk.
    assert client.post("/api/run/report_compare", json=body).status_code == 200
    assert wait_for_job(client, "report_compare")["status"] == "done"
    second = client.get("/api/state").get_json()["analysis"]["report_compare"]
    assert first["table"]["rows"] == second["table"]["rows"]


def test_unknown_report_stage_is_404(client):
    assert client.post("/api/run/report_nonsense", json={}).status_code == 404


def test_clear_caches_are_safe_to_call_repeatedly():
    for _ in range(2):
        clear_cache()
        clear_corpus_cache()
        clear_group_cache()
        clear_split_cache()


def test_load_raw_copy_flag(project):
    clear_cache()
    load_raw()
    assert load_raw(copy=False) is load_raw(copy=False)
    assert load_raw() is not load_raw()
