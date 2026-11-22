"""Clearing generated state.

This is the only code in the project that deletes anything, so the tests are
mostly about what it must *not* touch: the raw dataset unless asked, the
`.gitkeep` files that hold the directory layout in version control, files
that do not match a target's patterns, and anything a symlink points at
outside the project.
"""

from __future__ import annotations

import time

import pytest

from fraudulent_job_posting_detection import config, reset


@pytest.fixture
def populated(project):
    """A project with something in every output directory."""
    (config.PROCESSED_DIR / "corpus.parquet").write_bytes(b"x" * 64)
    (config.PROCESSED_DIR / "corpus_meta.json").write_text("{}")
    (config.INTERIM_DIR / "cv_results.csv").write_text("model,features\n")
    (config.MODELS_DIR / "ui_model.joblib").write_bytes(b"y" * 32)
    (config.MODELS_DIR / "ui_model.json").write_text("{}")
    (config.MODELS_DIR / "ui_results.json").write_text("{}")
    (config.FIGURES_DIR / "a.png").write_bytes(b"z" * 16)
    (config.TABLES_DIR / "t.csv").write_text("a,b")
    (config.TABLES_DIR / "t.md").write_text("|a|")
    for directory in (
        config.RAW_DIR, config.PROCESSED_DIR, config.INTERIM_DIR,
        config.MODELS_DIR, config.FIGURES_DIR, config.TABLES_DIR,
    ):
        (directory / ".gitkeep").touch()
    return project


# ---------------------------------------------------------------- describe --
def test_describe_lists_every_target(populated):
    summary = {item["key"]: item for item in reset.describe()}
    assert set(summary) == {
        "corpus", "interim", "models", "figures", "tables", "dataset"
    }
    assert summary["corpus"]["n_files"] == 2
    assert summary["models"]["n_files"] == 3
    assert summary["tables"]["n_files"] == 2


def test_only_the_dataset_is_guarded(populated):
    guarded = [item["key"] for item in reset.describe() if item["guarded"]]
    assert guarded == ["dataset"]


def test_describe_reports_sizes_and_examples(populated):
    corpus = next(i for i in reset.describe() if i["key"] == "corpus")
    assert corpus["bytes"] > 0
    assert "corpus.parquet" in corpus["examples"]


def test_describe_ignores_gitkeep(populated):
    for item in reset.describe():
        assert ".gitkeep" not in item["examples"]


def test_describe_on_an_empty_project(project):
    for item in reset.describe():
        if item["key"] == "dataset":
            continue  # the fixture writes a raw CSV
        assert item["n_files"] == 0


# ----------------------------------------------------------------- perform --
def test_reset_keeps_the_dataset_by_default(populated):
    result = reset.perform()
    assert result["dataset_removed"] is False
    assert config.RAW_CSV.exists()
    assert not (config.PROCESSED_DIR / "corpus.parquet").exists()
    assert not (config.MODELS_DIR / "ui_model.joblib").exists()
    assert not (config.FIGURES_DIR / "a.png").exists()
    assert not (config.TABLES_DIR / "t.csv").exists()


def test_reset_removes_the_dataset_only_when_asked(populated):
    assert reset.perform(include_dataset=True)["dataset_removed"] is True
    assert not config.RAW_CSV.exists()


def test_reset_preserves_gitkeep(populated):
    reset.perform(include_dataset=True)
    for directory in (
        config.RAW_DIR, config.PROCESSED_DIR, config.INTERIM_DIR,
        config.MODELS_DIR, config.FIGURES_DIR, config.TABLES_DIR,
    ):
        assert (directory / ".gitkeep").exists(), directory


def test_reset_leaves_files_outside_a_target_pattern(populated):
    """Figures are declared as *.png; a stray note is not generated output."""
    note = config.FIGURES_DIR / "notes.txt"
    note.write_text("hand-written")
    reset.perform()
    assert note.exists()


def test_reset_does_not_follow_symlinks_out_of_the_project(populated, tmp_path):
    """A symlink in an output directory must not be a way out."""
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive")
    link = config.MODELS_DIR / "escape.joblib"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform
        pytest.skip("symlinks unavailable")

    reset.perform()
    assert outside.exists()
    assert outside.read_text() == "must survive"


def test_reset_recreates_the_directory_tree(populated):
    reset.perform(include_dataset=True)
    for directory in (
        config.RAW_DIR, config.PROCESSED_DIR, config.INTERIM_DIR,
        config.MODELS_DIR, config.FIGURES_DIR, config.TABLES_DIR,
    ):
        assert directory.is_dir()


def test_reset_reports_what_it_removed(populated):
    result = reset.perform()
    assert result["n_removed"] >= 7
    assert result["bytes_freed"] > 0
    assert result["failed"] == []


def test_reset_is_idempotent(populated):
    first = reset.perform()
    second = reset.perform()
    assert first["n_removed"] > 0
    assert second["n_removed"] == 0


def test_reset_clears_the_caches(project):
    """A cached frame read from a file that no longer exists is a trap."""
    from fraudulent_job_posting_detection.data import load_raw
    from fraudulent_job_posting_detection.models import _CORPUS_CACHE, read_corpus
    from fraudulent_job_posting_detection.preprocess import run_preprocess

    run_preprocess()
    load_raw()
    read_corpus()
    assert _CORPUS_CACHE["frame"] is not None

    reset.perform()
    assert _CORPUS_CACHE["frame"] is None
    with pytest.raises(FileNotFoundError):
        read_corpus()


# --------------------------------------------------------------------- API --
def test_preview_endpoint(client):
    payload = client.get("/api/reset/preview").get_json()
    assert payload["ok"] is True
    assert {item["key"] for item in payload["targets"]} >= {"corpus", "dataset"}
    assert payload["active_job"] is None


def test_reset_endpoint_keeps_the_dataset(client, populated):
    response = client.post("/api/reset", json={})
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert "kept" in body["message"]
    assert config.RAW_CSV.exists()


def test_reset_endpoint_can_remove_the_dataset(client, populated):
    response = client.post("/api/reset", json={"include_dataset": True})
    assert response.status_code == 200
    assert not config.RAW_CSV.exists()


def test_reset_endpoint_defaults_to_keeping_the_dataset(client, populated):
    """A request with no body must not take the destructive branch."""
    assert client.post("/api/reset").status_code == 200
    assert config.RAW_CSV.exists()


def test_reset_is_refused_while_a_job_runs(client, monkeypatch):
    """Deleting under a running worker leaves half-written files."""
    from fraudulent_job_posting_detection.webapp import jobs

    monkeypatch.setattr(jobs, "active", lambda: "train")
    response = client.post("/api/reset", json={})
    assert response.status_code == 409
    assert "still running" in response.get_json()["error"]


def test_preview_surfaces_a_running_job(client, monkeypatch):
    from fraudulent_job_posting_detection.webapp import jobs

    monkeypatch.setattr(jobs, "active", lambda: "clean")
    assert client.get("/api/reset/preview").get_json()["active_job"] == "clean"


def test_reset_clears_the_job_registry(client, project):
    """Leaving 'clean: done' beside a deleted corpus describes a state that
    no longer exists."""
    from fraudulent_job_posting_detection.webapp import jobs

    assert client.post("/api/run/clean", json={"options": {}}).status_code == 200
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if jobs.active() is None:
            break
        time.sleep(0.2)
    assert jobs.snapshots().get("clean", {}).get("status") == "done"

    assert client.post("/api/reset", json={}).status_code == 200
    assert jobs.snapshots() == {}


def test_state_is_empty_after_a_reset(client, project):
    from fraudulent_job_posting_detection.preprocess import run_preprocess

    run_preprocess()
    assert client.get("/api/state").get_json()["corpus_ready"] is True
    client.post("/api/reset", json={})
    state = client.get("/api/state").get_json()
    assert state["corpus_ready"] is False
    assert state["model"]["name"] is None
    assert state["results"] is None
    assert state["analysis"] == {}
    # The dataset survived, so the pipeline can be run again immediately.
    assert state["dataset"] is not None
