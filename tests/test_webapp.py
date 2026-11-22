"""The HTTP surface: state, uploads, jobs, prediction, and figure serving.

The upload tests are the important ones. The previous implementation wrote
the uploaded file straight over the working dataset and validated afterwards,
so uploading the wrong file destroyed a dataset that is not redistributable
and cannot simply be re-downloaded without a Kaggle session.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import pytest

from fraudulent_job_posting_detection import config


def wait_for_job(client, stage, timeout=300):
    """Block until `stage` finishes.

    Fails immediately if the stage was never started, rather than polling
    until the timeout: a renamed or rejected stage would otherwise hang the
    whole suite instead of failing with a useful message.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get("/api/jobs").get_json()
        job = payload["jobs"].get(stage)
        if job is None and payload.get("active") is None:
            raise AssertionError(f"job {stage} was never started")
        if job and job.get("status") in {"done", "error", "cancelled"}:
            return job
        if job and job.get("status") == "idle" and payload.get("active") is None:
            raise AssertionError(f"job {stage} is idle; it never started")
        time.sleep(0.2)
    raise AssertionError(f"job {stage} did not finish within {timeout}s")


# ------------------------------------------------------------------- page --
def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Fraudulent job posting detection" in response.data


def test_state_describes_a_fresh_project(client):
    state = client.get("/api/state").get_json()
    assert state["dataset"]["rows"] > 0
    assert state["corpus_ready"] is False
    assert state["model"]["name"] is None
    assert state["active_job"] is None


def test_state_is_json_serialisable_end_to_end(client):
    """numpy scalars from pandas are not JSON-serialisable by default."""
    import json

    json.dumps(client.get("/api/state").get_json())


# ----------------------------------------------------------------- upload --
def test_upload_rejects_a_non_csv(client):
    response = client.post(
        "/api/upload",
        data={"csv": (io.BytesIO(b"not a csv"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "csv" in response.get_json()["error"].lower()


def test_upload_rejects_an_empty_request(client):
    response = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_invalid_upload_leaves_the_existing_dataset_intact(client, project):
    """The regression that matters: a bad upload must not destroy good data."""
    before = project["RAW_CSV"].read_bytes()

    junk = pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(index=False).encode()
    response = client.post(
        "/api/upload",
        data={"csv": (io.BytesIO(junk), "wrong.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "kept" in response.get_json()["error"].lower()
    assert project["RAW_CSV"].read_bytes() == before
    assert client.get("/api/state").get_json()["dataset"] is not None


def test_failed_upload_leaves_no_staging_file(client, project):
    junk = pd.DataFrame({"a": [1]}).to_csv(index=False).encode()
    client.post(
        "/api/upload",
        data={"csv": (io.BytesIO(junk), "wrong.csv")},
        content_type="multipart/form-data",
    )
    leftovers = list(project["RAW_DIR"].glob("*.incoming"))
    assert leftovers == []


def test_valid_upload_replaces_the_dataset(client, project, raw_frame):
    smaller = raw_frame.iloc[:120]
    payload = smaller.to_csv(index=False).encode()
    response = client.post(
        "/api/upload",
        data={"csv": (io.BytesIO(payload), "fake_job_postings.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_json()
    assert client.get("/api/state").get_json()["dataset"]["rows"] == 120


# ------------------------------------------------------------------- jobs --
def test_unknown_stage_is_404(client):
    assert client.post("/api/run/wat", json={}).status_code == 404
    assert client.post("/api/cancel/wat", json={}).status_code == 404


def test_cancelling_an_idle_job_is_409(client):
    assert client.post("/api/cancel/train", json={}).status_code == 409


def test_training_without_a_corpus_is_rejected(client):
    response = client.post("/api/run/train", json={"models": ["LinearSVC"]})
    assert response.status_code == 400
    assert "corpus" in response.get_json()["error"].lower()


def test_training_with_no_models_is_rejected(client, project):
    client.post("/api/run/clean", json={"options": {}})
    wait_for_job(client, "clean")
    response = client.post("/api/run/train", json={"models": []})
    assert response.status_code == 400


def test_clean_job_runs_and_reports_progress(client):
    assert client.post("/api/run/clean", json={"options": {}}).status_code == 200
    job = wait_for_job(client, "clean")
    assert job["status"] == "done", job["error"]
    assert job["percent"] == 100.0
    state = client.get("/api/state").get_json()
    assert state["corpus_ready"] is True
    assert state["provenance"] is not None


def test_clean_job_honours_submitted_options(client):
    client.post("/api/run/clean", json={"options": {"stem": False}})
    assert wait_for_job(client, "clean")["status"] == "done"
    provenance = client.get("/api/state").get_json()["provenance"]
    flags = {option["key"]: option["enabled"] for option in provenance["options"]}
    assert flags["stem"] is False


def test_only_one_job_runs_at_a_time(client):
    """Two jobs at once contend for memory and for pyplot's global state."""
    assert client.post("/api/run/clean", json={"options": {}}).status_code == 200
    second = client.post("/api/run/report_eda", json={})
    assert second.status_code == 409
    assert "already running" in second.get_json()["error"]
    wait_for_job(client, "clean")


def test_report_eda_job_produces_figures(client):
    assert client.post("/api/run/report_eda", json={}).status_code == 200
    job = wait_for_job(client, "report_eda")
    assert job["status"] == "done", job["error"]
    figures = client.get("/api/state").get_json()["analysis_figures"]
    assert figures
    assert all(item["name"].endswith(".png") for item in figures)


# ---------------------------------------------------------------- figures --
def test_figure_route_serves_a_generated_png(client):
    client.post("/api/run/report_eda", json={})
    wait_for_job(client, "report_eda")
    name = client.get("/api/state").get_json()["analysis_figures"][0]["name"]
    response = client.get(f"/figure/{name}")
    assert response.status_code == 200
    assert response.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_figure_route_refuses_non_png(client):
    assert client.get("/figure/secrets.txt").status_code == 404


def test_figure_route_refuses_traversal(client):
    response = client.get("/figure/../../../../etc/passwd")
    assert response.status_code in {400, 403, 404}


# ---------------------------------------------------------------- predict --
def test_predict_without_a_model_is_409(client):
    response = client.post("/api/predict", json={"description": "anything"})
    assert response.status_code == 409


@pytest.mark.slow
def test_predict_round_trip(client):
    client.post("/api/run/clean", json={"options": {}})
    assert wait_for_job(client, "clean")["status"] == "done"
    client.post("/api/run/train", json={"models": ["LinearSVC"], "strategy": "none"})
    job = wait_for_job(client, "train")
    assert job["status"] == "done", job["error"]

    response = client.post("/api/predict", json={
        "title": "Data Entry Clerk",
        "description": "Earn money fast working from home, no experience necessary.",
        "has_company_logo": False,
    })
    assert response.status_code == 200
    result = response.get_json()["result"]
    assert result["label"] in {"FRAUDULENT", "LEGITIMATE"}
    assert isinstance(result["score"], float)
    assert result["score_kind"] in {"margin", "probability"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["cleaned_text"]
    # One entry per binary flag, model-agnostic.
    assert len(result["sensitivity"]) == len(config.BINARY_FLAGS)


@pytest.mark.slow
def test_predict_requires_a_description(client):
    client.post("/api/run/clean", json={"options": {}})
    wait_for_job(client, "clean")
    client.post("/api/run/train", json={"models": ["LinearSVC"], "strategy": "none"})
    wait_for_job(client, "train")
    response = client.post("/api/predict", json={"description": "   "})
    assert response.status_code == 400


@pytest.mark.slow
def test_prediction_is_escaped_not_executed(client):
    """Dataset text is untrusted; it must come back as data."""
    client.post("/api/run/clean", json={"options": {}})
    wait_for_job(client, "clean")
    client.post("/api/run/train", json={"models": ["LinearSVC"], "strategy": "none"})
    wait_for_job(client, "train")
    response = client.post("/api/predict", json={
        "description": "<script>alert(1)</script> earn money fast",
    })
    assert response.status_code == 200
    assert "<script>" not in response.get_json()["result"]["cleaned_text"]


@pytest.mark.slow
def test_training_results_survive_a_restart(client, project):
    """A saved model with no results beside it is useless."""
    from fraudulent_job_posting_detection.webapp import app as webapp

    client.post("/api/run/clean", json={"options": {}})
    wait_for_job(client, "clean")
    client.post("/api/run/train", json={"models": ["LinearSVC"], "strategy": "none"})
    wait_for_job(client, "train")

    fresh = webapp.create_app()
    fresh.config["TESTING"] = True
    state = fresh.test_client().get("/api/state").get_json()
    assert state["results"] is not None
    assert state["model"]["name"] == "LinearSVC"
