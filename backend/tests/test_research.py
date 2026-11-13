import json
import time
import numpy as np
import pytest
from flask import Flask

from research import datasets, correlation, causal, validation, stats as S
from research.api import init_research
from research.llm import Reasoner, template_hypotheses
from research.store import to_jsonable

EX = template_hypotheses({"x_label": "x", "y_label": "y"})


def _run(preset):
    df = datasets.generate(preset)
    c = correlation.run_correlation(df, "x", "y")
    return df, c, causal.run_causality(df, c, EX), validation.run_validation(df)


def make_client(tmp_path, **kw):
    app = Flask(__name__)
    init_research(app, str(tmp_path / "t.db"), **kw)
    return app.test_client()


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path, sync=True, reasoner=Reasoner(api_key=""))


def post(client, preset):
    r = client.post("/api/research/runs", json={"preset_id": preset})
    assert r.status_code == 202
    return client.get(f"/api/research/runs/{r.get_json()['run_id']}").get_json()


# ---- math
def test_pearson_matches_numpy_and_ci_contains_r():
    df = datasets.generate("demo_confounder")
    c = correlation.run_correlation(df, "x", "y")
    assert c["r"] == pytest.approx(np.corrcoef(df.x, df.y)[0, 1])
    assert c["ci_95"][0] < c["r"] < c["ci_95"][1]
    assert c["n"] == 750


def test_holm_known_values():
    adj, rej = S.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06]) and rej == [True, False, False]


def test_lagged_pair_alignment():
    x, y = np.arange(10.0), np.arange(10.0) * 10
    a, b = S.lagged_pair(x, y, 2)
    assert a[0] == 0 and b[0] == 20 and len(a) == len(b) == 8


def test_datasets_seeded_and_distinct():
    a, b = datasets.generate("demo_effect", 1), datasets.generate("demo_effect", 1)
    assert a.equals(b) and not a.equals(datasets.generate("demo_effect", 2))


# ---- scenarios (criteria fixed in research/criteria.py before running)
def test_null():
    _, c, ca, v = _run("demo_null")
    assert c["association_verdict"] == "not_detected"
    assert ca["causal_verdict"] == "not_established"
    assert v["reliability_verdict"] == "rejected"


def test_confounder_detected_but_not_causal():
    _, c, ca, v = _run("demo_confounder")
    assert c["association_verdict"] == "detected"
    assert ca["causal_verdict"] == "contradicted"
    res = {t["id"]: t["result"] for t in ca["tests"]}
    assert res["confounder_control"] == "fail"
    assert v["reliability_verdict"] == "supported"  # stable association != causation


def test_planted_effect():
    _, c, ca, v = _run("demo_effect")
    assert c["association_verdict"] == "detected"
    assert ca["causal_verdict"] == "evidence_consistent"
    assert ca["selected_lag"] == 1 and v["selected_lag"] == 1
    assert v["reliability_verdict"] == "supported"


def test_null_false_positive_rate_over_seeds():
    fp = 0
    for seed in range(100, 130):
        df = datasets.generate("demo_null", seed)
        c = correlation.run_correlation(df, "x", "y")
        v = validation.run_validation(df)
        fp += (c["association_verdict"] == "detected") or v["reliability_verdict"] == "supported"
    assert fp <= 3  # of 30 (nominal ~5%)


def test_no_future_leakage_in_lag_selection():
    df = datasets.generate("demo_effect")
    a, _ = validation.split_indices(len(df))
    k = validation.select_lag_on_train(df.iloc[:a])
    df2 = df.copy()
    df2.iloc[a:, df2.columns.get_loc("y")] = np.random.default_rng(0).normal(size=len(df) - a)
    assert validation.select_lag_on_train(df2.iloc[:a]) == k
    v2 = validation.run_validation(df2)
    assert v2["selected_lag"] == k
    assert v2["reliability_verdict"] != "supported"  # corrupted future must not validate


def test_short_sample_inconclusive():
    df = datasets.generate("demo_effect", n=100)
    assert validation.run_validation(df)["reliability_verdict"] == "inconclusive"


def test_to_jsonable_strict():
    out = to_jsonable({"a": np.float64("nan"), "b": np.int64(3), "c": [np.inf], "d": np.bool_(True)})
    assert out == {"a": None, "b": 3, "c": [None], "d": True}
    json.dumps(out, allow_nan=False)


# ---- API
def test_health_and_presets(client):
    assert client.get("/api/health").get_json()["status"] == "ok"
    ids = [p["id"] for p in client.get("/api/research/presets").get_json()["presets"]]
    assert ids == ["demo_null", "demo_confounder", "demo_effect"]


@pytest.mark.parametrize("body", [{"preset_id": "AAPL"}, {}, {"preset_id": "demo_null", "code": "x"}, {"preset_id": 3}])
def test_bad_input_rejected(client, body):
    r = client.post("/api/research/runs", json=body)
    assert r.status_code == 400 and "error" in r.get_json()


def test_non_json_rejected(client):
    assert client.post("/api/research/runs", data="nope").status_code == 400


def test_unknown_run_404(client):
    assert client.get("/api/research/runs/nope").status_code == 404


@pytest.mark.parametrize("preset,causal_v", [("demo_null", "not_established"),
                                              ("demo_confounder", "contradicted"),
                                              ("demo_effect", "evidence_consistent")])
def test_full_report(client, preset, causal_v):
    rep = post(client, preset)
    assert rep["status"] == "completed" and rep["current_stage"] == "complete"
    assert [s["status"] for s in rep["stages"]] == ["completed"] * 3
    assert rep["causality"]["causal_verdict"] == causal_v
    assert rep["provenance"]["reasoning_source"] == "template_demo"
    assert rep["ground_truth"]["verdict_matches_expected"] is True
    assert len(rep["visualization_data"]["aligned_series"]) == 750
    rr = rep["visualization_data"]["rolling_correlation"]
    assert rr[0]["r"] is None and rr[-1]["r"] is not None
    json.dumps(rep, allow_nan=False)
    assert "GEMINI" not in json.dumps(rep)
    assert rep["error"] is None
    assert client.get("/api/research/runs").get_json()["runs"]


def test_ground_truth_not_sent_to_reasoner(tmp_path):
    seen = {}

    class Spy(Reasoner):
        def hypotheses(self, ctx):
            seen.update(ctx)
            return super().hypotheses(ctx)
    c = make_client(tmp_path, sync=True, reasoner=Spy(api_key=""))
    post(c, "demo_confounder")
    blob = json.dumps(seen).lower()
    assert seen and "confound" not in blob and "ground" not in blob and "0.8" not in blob


# ---- Gemini paths (fake client, no network)
class _Resp:
    def __init__(self, text):
        self.text = text


class FakeClient:
    def __init__(self, texts=None, exc=None):
        self.texts, self.exc = list(texts or []), exc
        self.models = self

    def generate_content(self, **kw):
        if self.exc:
            raise self.exc
        return _Resp(self.texts.pop(0))


GOOD_H = json.dumps({"candidate_explanations": [
    {"id": "a", "title": "t", "type": "direct_effect", "description": "d"},
    {"id": "b", "title": "t", "type": "common_cause", "description": "d"}],
    "possible_confounders": [{"name": "m", "rationale": "r", "observed": True}]})


def test_gemini_provider_failure_falls_back_visibly(tmp_path):
    c = make_client(tmp_path, sync=True, reasoner=Reasoner(client=FakeClient(exc=RuntimeError("quota"))))
    rep = post(c, "demo_confounder")
    assert rep["status"] == "completed"  # statistics unaffected
    assert rep["provenance"]["reasoning_source"] == "template_fallback"
    assert any("Gemini" in w for w in rep["warnings"])


def test_gemini_valid_hypotheses_used_but_invented_number_rejected(tmp_path):
    bad = json.dumps({"causality_reasoning": "r was 0.999 so causal", "validation_reasoning": "ok"})
    c = make_client(tmp_path, sync=True, reasoner=Reasoner(client=FakeClient([GOOD_H, bad])))
    rep = post(c, "demo_null")
    assert rep["provenance"]["reasoning_source"] == "template_fallback"
    assert rep["causality"]["candidate_explanations"][0]["id"] == "a"


def test_gemini_fully_valid_labelled_gemini(tmp_path):
    ok = json.dumps({"causality_reasoning": "No robust association; causation not established.",
                     "validation_reasoning": "The sceptic rejected the claim."})
    c = make_client(tmp_path, sync=True, reasoner=Reasoner(client=FakeClient([GOOD_H, ok])))
    assert post(c, "demo_null")["provenance"]["reasoning_source"] == "gemini"


def test_gemini_schema_violation_rejected(tmp_path):
    c = make_client(tmp_path, sync=True, reasoner=Reasoner(client=FakeClient(['{"nope":1}', '{"nope":1}'])))
    assert post(c, "demo_null")["provenance"]["reasoning_source"] == "template_fallback"


def test_pipeline_internal_failure_reported(tmp_path, monkeypatch):
    from research import pipeline
    monkeypatch.setattr(pipeline.validation, "run_validation", lambda df: 1 / 0)
    rep = post(make_client(tmp_path, sync=True, reasoner=Reasoner(api_key="")), "demo_null")
    assert rep["status"] == "failed" and rep["error"]["stage"] == "validation"
    assert rep["correlation"] is not None  # partial results retained


def test_timeout_marks_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_JOB_TIMEOUT", "0")
    rep = post(make_client(tmp_path, sync=True, reasoner=Reasoner(api_key="")), "demo_null")
    assert rep["status"] == "failed" and rep["error"]["code"] == "timeout"


def test_threaded_mode_polls_to_completion(tmp_path):
    c = make_client(tmp_path, sync=False, reasoner=Reasoner(api_key=""))
    rid = c.post("/api/research/runs", json={"preset_id": "demo_effect"}).get_json()["run_id"]
    seen = []
    for _ in range(300):
        rep = c.get(f"/api/research/runs/{rid}").get_json()
        seen.append(rep["current_stage"])
        if rep["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert rep["status"] == "completed"
    order = [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]]
    assert order == [s for s in ["queued", "correlation", "causality", "validation", "complete"] if s in order]


def test_items_api_and_health_in_real_app():
    import app as appmod
    c = appmod.app.test_client()
    assert c.get("/api/items").status_code == 200
    assert c.get("/api/health").status_code == 200
