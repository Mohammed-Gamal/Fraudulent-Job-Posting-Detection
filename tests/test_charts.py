"""Chart payloads.

The browser draws whatever these builders return, so the shape of the payload
is the contract. A missing key or a NaN reaches the page as a blank chart or
an exception in the console, neither of which is visible server-side.
"""

from __future__ import annotations

import json
import math

import pytest

from fraudulent_job_posting_detection import charts, config

ALL_KEYS = [spec.key for spec in charts.CHART_SPECS]


@pytest.fixture(autouse=True)
def _fresh_cache():
    charts.clear_cache()
    yield
    charts.clear_cache()


def _finite(value) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


# ------------------------------------------------------------- catalogue --
def test_catalogue_describes_every_chart():
    entries = charts.catalogue()
    assert len(entries) == len(charts.CHART_SPECS)
    for entry in entries:
        assert {"key", "label", "blurb", "kind", "default", "options"} <= set(entry)
        assert entry["kind"]


def test_validate_keys_drops_unknown_and_keeps_order():
    assert charts.validate_keys(["category", "length", "nope"]) == [
        key for key in ALL_KEYS if key in {"category", "length"}
    ]
    assert charts.validate_keys([]) == []
    assert charts.validate_keys(["nope"]) == []


# ---------------------------------------------------------------- shapes --
@pytest.mark.parametrize("key", ALL_KEYS)
def test_every_chart_builds_and_is_json_safe(project, key):
    payload = charts.build([key])
    assert key in payload
    spec = payload[key]
    assert "error" not in spec, spec.get("error")
    assert spec["kind"]
    assert spec["title"]
    # The page serialises this straight to the browser.
    json.dumps(spec)
    assert _finite(spec)


def test_histogram_series_align_with_the_bin_edges(project):
    spec = charts.build(["length"])["length"]
    bins = len(spec["edges"]) - 1
    assert len(spec["series"]) == 2
    for series in spec["series"]:
        assert len(series["density"]) == bins
        assert len(series["counts"]) == bins
        # Densities are shares of that class, so they sum to one.
        assert 0.0 <= sum(series["density"]) <= 1.0001
        assert series["n"] > 0


def test_histogram_labels_are_the_two_classes(project):
    spec = charts.build(["length"])["length"]
    assert {s["label"] for s in spec["series"]} == {"legitimate", "fraudulent"}


def test_flags_chart_covers_every_binary_flag(project):
    spec = charts.build(["flags"])["flags"]
    assert [g["label"] for g in spec["groups"]] == list(config.BINARY_FLAGS)
    for group in spec["groups"]:
        assert group["bars"]
        for bar in group["bars"]:
            assert 0 <= bar["rate"] <= 100
            assert bar["n"] >= bar["n_fraud"] >= 0
    assert 0 <= spec["baseline"] <= 100


def test_terms_on_each_side_agree_with_that_side_s_claim(project):
    """A term is only shown as indicative of a class if its sign says so.

    The ranking takes the top N, which on a small corpus can return "most
    fraudulent" terms that still lean legitimate. Labelling those as fraud
    indicators would be a false claim, so they are dropped instead.
    """
    spec = charts.build(["terms"])["terms"]
    assert all(r["log_odds"] > 0 for r in spec["fraudulent"])
    assert all(r["log_odds"] < 0 for r in spec["legitimate"])
    assert spec["fraudulent"] or spec["legitimate"]


def test_terms_chart_is_sorted_outward_from_zero(project):
    spec = charts.build(["terms"])["terms"]
    fraud = [r["log_odds"] for r in spec["fraudulent"]]
    legit = [r["log_odds"] for r in spec["legitimate"]]
    assert fraud == sorted(fraud, reverse=True)
    assert legit == sorted(legit)


def test_terms_normalisation_is_switchable(project):
    normalised = charts.build(["terms"], {"terms": "normalised"})["terms"]
    raw = charts.build(["terms"], {"terms": "raw"})["terms"]
    assert normalised["normalised"] is True
    assert raw["normalised"] is False


def test_missingness_rows_are_percentages(project):
    spec = charts.build(["missingness"])["missingness"]
    assert spec["rows"]
    for row in spec["rows"]:
        assert 0 <= row["legitimate"] <= 100
        assert 0 <= row["fraudulent"] <= 100
        assert row["difference"] == pytest.approx(
            row["fraudulent"] - row["legitimate"], abs=0.02
        )


def test_category_chart_honours_the_requested_field(project):
    for field in config.CATEGORICAL_FIELDS[:3]:
        spec = charts.build(["category"], {"category": field})["category"]
        assert spec["field"] == field
        assert field.replace("_", " ") in spec["title"]


def test_category_chart_falls_back_on_an_unknown_field(project):
    spec = charts.build(["category"], {"category": "not_a_column"})["category"]
    assert spec["field"] in config.CATEGORICAL_FIELDS


def test_category_rows_are_ranked_and_thresholded(project):
    spec = charts.build(["category"], {"category": "industry"})["category"]
    rates = [row["rate"] for row in spec["rows"]]
    assert rates == sorted(rates, reverse=True)
    # fraud_rate_by drops anything under 30 postings as noise.
    assert all(row["n"] >= 30 for row in spec["rows"])


# ----------------------------------------------------------------- cache --
def test_results_are_cached_per_option(project):
    first = charts.build(["category"], {"category": "industry"})["category"]
    again = charts.build(["category"], {"category": "industry"})["category"]
    assert first is again
    other = charts.build(["category"], {"category": "function"})["category"]
    assert other is not first


def test_cache_can_be_cleared(project):
    first = charts.build(["flags"])["flags"]
    charts.clear_cache()
    assert charts.build(["flags"])["flags"] is not first


def test_building_nothing_returns_nothing(project):
    assert charts.build([]) == {}
    assert charts.build(["unknown"]) == {}


def test_a_failing_builder_is_reported_not_raised(project, monkeypatch):
    def explode(frame, option):
        raise RuntimeError("boom")

    monkeypatch.setitem(charts.BUILDERS, "flags", explode)
    payload = charts.build(["flags"])
    assert "error" in payload["flags"]


# ------------------------------------------------------------------- API --
def test_chart_endpoint_returns_requested_charts(client):
    response = client.post(
        "/api/explore/charts", json={"charts": ["flags", "length"]}
    )
    assert response.status_code == 200
    assert set(response.get_json()["charts"]) == {"flags", "length"}


def test_chart_endpoint_ignores_unknown_keys(client):
    response = client.post("/api/explore/charts", json={"charts": ["nope"]})
    assert response.status_code == 200
    assert response.get_json()["charts"] == {}


def test_chart_endpoint_tolerates_a_junk_options_field(client):
    response = client.post(
        "/api/explore/charts", json={"charts": ["flags"], "options": "not-a-dict"}
    )
    assert response.status_code == 200


def test_chart_endpoint_needs_a_dataset(client, project):
    project["RAW_CSV"].unlink()
    from fraudulent_job_posting_detection.data import clear_cache

    clear_cache()
    response = client.post("/api/explore/charts", json={"charts": ["flags"]})
    assert response.status_code == 409
