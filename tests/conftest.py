"""Shared fixtures.

Every test runs against a synthetic corpus in a temporary directory, never
the real 50 MB EMSCAD file. That keeps the suite fast enough to run on every
change, and means it passes on a clean checkout where `data/raw/` is empty --
the dataset is not redistributable, so a suite that needed it would be a
suite nobody could run.

`config` holds module-level paths, so the fixture redirects them and restores
them afterwards. Tests that touch the filesystem must request `project`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudulent_job_posting_detection import config

# Deterministic fixture data: a fixed seed, so a failure is reproducible.
SEED = 20260101

FRAUD_PHRASES = [
    "earn money fast working from home no experience necessary",
    "urgent hiring data entry clerk immediate start weekly pay",
    "apply using the link below and send your details to our recruiter",
    "work from home earn cash no experience required start today",
]
LEGIT_PHRASES = [
    "we are seeking an experienced software engineer for our platform team",
    "the successful candidate will design distributed systems and analytics pipelines",
    "responsibilities include maintaining infrastructure and mentoring junior staff",
    "our consultancy has operated across europe since nineteen ninety eight",
]


def _make_frame(n_legit: int = 160, n_fraud: int = 40, duplicates: int = 12):
    """A small frame with the structure that matters: imbalance and duplicates."""
    rng = np.random.default_rng(SEED)
    rows = []
    job_id = 1

    for index in range(n_legit + n_fraud):
        is_fraud = index >= n_legit
        phrases = FRAUD_PHRASES if is_fraud else LEGIT_PHRASES
        description = " ".join(rng.choice(phrases, size=2, replace=True))
        rows.append({
            "job_id": job_id,
            "title": "Data Entry Clerk" if is_fraud else "Software Engineer",
            "location": "GB, LND, London",
            "department": "Engineering",
            "salary_range": None,
            "company_profile": None if is_fraud and index % 2 == 0 else "An established firm.",
            "description": f"{description} record {index}",
            "requirements": (
                "A computer and internet access." if is_fraud
                else "Five years of commercial experience."
            ),
            "benefits": None if is_fraud else "Pension and healthcare.",
            "telecommuting": int(is_fraud),
            "has_company_logo": 0 if is_fraud else 1,
            "has_questions": 0 if is_fraud else 1,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology",
            "function": "Engineering",
            "fraudulent": int(is_fraud),
        })
        job_id += 1

    frame = pd.DataFrame(rows)

    # Repeat one description across several postings so the grouped split has
    # something real to keep together.
    shared = "a shared duplicated description used by several postings"
    for position in range(duplicates):
        frame.loc[position, "description"] = shared
    return frame


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return _make_frame()


@pytest.fixture
def project(tmp_path: Path, monkeypatch, raw_frame: pd.DataFrame):
    """Point every config path at a temp directory holding a synthetic dataset."""
    from fraudulent_job_posting_detection import data

    paths = {
        "PROJECT_ROOT": tmp_path,
        "DATA_DIR": tmp_path / "data",
        "RAW_DIR": tmp_path / "data" / "raw",
        "INTERIM_DIR": tmp_path / "data" / "interim",
        "PROCESSED_DIR": tmp_path / "data" / "processed",
        "REPORTS_DIR": tmp_path / "reports",
        "FIGURES_DIR": tmp_path / "reports" / "figures",
        "TABLES_DIR": tmp_path / "reports" / "tables",
        "MODELS_DIR": tmp_path / "models",
        "RAW_CSV": tmp_path / "data" / "raw" / "fake_job_postings.csv",
    }
    for name, value in paths.items():
        monkeypatch.setattr(config, name, value, raising=True)

    # Modules that captured a path at import time need the same treatment.
    from fraudulent_job_posting_detection import models, preprocess

    monkeypatch.setattr(
        preprocess, "CORPUS_PATH", paths["PROCESSED_DIR"] / "corpus.parquet"
    )
    monkeypatch.setattr(
        preprocess, "CORPUS_META_PATH", paths["PROCESSED_DIR"] / "corpus_meta.json"
    )
    monkeypatch.setattr(models, "RESULTS_PATH", paths["INTERIM_DIR"] / "cv_results.csv")
    monkeypatch.setattr(models, "UI_MODEL_PATH", paths["MODELS_DIR"] / "ui_model.joblib")
    monkeypatch.setattr(models, "UI_MODEL_META_PATH", paths["MODELS_DIR"] / "ui_model.json")

    from fraudulent_job_posting_detection import analysis

    monkeypatch.setattr(
        analysis, "ANALYSIS_PATH", paths["MODELS_DIR"] / "ui_analysis.json"
    )

    # These caches key on file identity, and a temp project reuses paths
    # across tests fast enough that a stale entry can survive.
    from fraudulent_job_posting_detection import charts
    from fraudulent_job_posting_detection.splits import (
        clear_group_cache,
        clear_split_cache,
    )

    models.clear_corpus_cache()
    clear_group_cache()
    clear_split_cache()
    charts.clear_cache()

    # The fixture frame is deliberately not 17,880 rows, so relax the check.
    monkeypatch.setattr(config, "EXPECTED_ROWS", len(raw_frame))

    config.ensure_directories()
    raw_frame.to_csv(paths["RAW_CSV"], index=False)
    data.clear_cache()
    yield paths
    data.clear_cache()
    models.clear_corpus_cache()
    clear_group_cache()
    clear_split_cache()
    charts.clear_cache()


@pytest.fixture
def corpus(project):
    """A built corpus in the temp project."""
    from fraudulent_job_posting_detection.preprocess import run_preprocess

    return run_preprocess()


@pytest.fixture(autouse=True)
def fresh_job_registry():
    """Give every test an empty job registry.

    The registry is a module-level singleton, which is correct for a
    single-user local app but means state leaks between tests: a job left
    running at the end of one test makes the next test's `start` return 409,
    and a finished job from an earlier test makes a poll return "done"
    immediately for work that never ran in this one.
    """
    from fraudulent_job_posting_detection.webapp import jobs

    jobs._REGISTRY = jobs._Registry()
    yield
    jobs._REGISTRY = jobs._Registry()


@pytest.fixture
def client(project, monkeypatch):
    """A Flask test client wired to the temp project."""
    from fraudulent_job_posting_detection.webapp import app as webapp

    monkeypatch.setattr(webapp, "RESULTS_PATH", project["MODELS_DIR"] / "ui_results.json")
    application = webapp.create_app()
    application.config["TESTING"] = True
    return application.test_client()


def pytest_configure(config):  # noqa: A002 - pytest fixes this hook's argument name
    config.addinivalue_line(
        "markers", "slow: trains a model; excluded by default with -m 'not slow'"
    )


@pytest.fixture(scope="session")
def torch_or_skip():
    from fraudulent_job_posting_detection.embeddings import torch_available

    if not torch_available():
        pytest.skip("PyTorch is not installed")
    return True
