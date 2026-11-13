"""Autonomous / guided / demo research engine. All offline: Gemini is a fake client,
FRED is a deterministic stub, Alpaca is unconfigured unless stated."""
import json

import numpy as np
import pandas as pd
import pytest
from flask import Flask

from research import hstages as H, registry as R, stats as S, transforms as T, validation as V
from research.api import init_research
from research.hypothesis import HypothesisProposal, validate_proposal
from research.llm import Reasoner
from research.providers import AlpacaProvider, DataService, FixtureProvider, ProviderError, _get


# ------------------------------------------------------------------ helpers
class StubFred:
    """Deterministic fake FRED: independent random walks on a business-day grid."""
    mode = "stub"
    calls = 0

    def __init__(self, fail=()):
        self.fail = set(fail)

    def fetch(self, v, start, end):
        StubFred.calls += 1
        if v.id in self.fail:
            raise ProviderError(f"stub failure for {v.id}")
        idx = pd.bdate_range(start, end)
        rng = np.random.default_rng(sum(map(ord, v.id)))
        r = rng.standard_normal(len(idx))
        if v.id == "SP500":                                   # planted: yesterday's yield change lowers today's return
            d = np.random.default_rng(sum(map(ord, "DGS10"))).standard_normal(len(idx))
            r = r - 1.0 * np.r_[0, d[:-1]]
        s = (3 + np.cumsum(r) * 0.02) if v.kind in ("rate", "spread") else 100 * np.exp(np.cumsum(r) / 100)
        return pd.Series(s, index=idx), {"provider": "stub", "series_id": v.id}


class FakeGemini:
    def __init__(self, generation=None, causal=None, narrative=None, boom=None, ranking=None):
        self.generation, self.causal, self.narrative, self.boom, self.ranking = generation, causal, narrative, boom, ranking
        self.models = self
        self.prompts = []

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        if self.boom:
            raise RuntimeError(self.boom)
        if "autonomous quantitative research agent" in contents:
            out = self.generation
        elif "sceptical econometrician" in contents:
            out = self.causal
        elif "VIABLE and REALISTIC" in contents:
            out = self.ranking
        else:
            out = self.narrative
        if out is None:
            raise RuntimeError("no scripted response")
        return type("R", (), {"text": out if isinstance(out, str) else json.dumps(out)})()


def hyp(**kw):
    base = {"question": "Do changes in the 10-year yield predict next-day S&P 500 returns?",
            "predictor_variable": "DGS10", "predictor_transform": "diff", "outcome_variable": "SP500",
            "outcome_transform": "log_return", "direction": "negative", "lag_days": 1,
            "null_hypothesis": "Yield changes carry no information about next-day returns.",
            "mechanism": "Higher discount rates reduce equity valuations.",
            "competing_explanations": ["Common market shocks"], "control_variables": ["DTWEXBGS"],
            "proposed_test": "lagged_correlation_hac", "limitations": ["linear only"]}
    base.update(kw)
    return base


def gen(*hs, unsupported=""):
    return {"hypotheses": list(hs), "unsupported_reason": unsupported}


CAUSAL = {"mechanisms": ["rates affect discounting"], "alternative_explanations": ["common shocks", "reverse causation"],
          "confounders": [{"variable_id": "DTWEXBGS", "rationale": "dollar moves with both"}],
          "dag_edges": [{"source": "DGS10", "target": "SP500", "rationale": "discounting"},
                        {"source": "latent:growth_news", "target": "DGS10", "rationale": "news"},
                        {"source": "EVIL;DROP", "target": "SP500", "rationale": "bad node"}],
          "assumptions": ["linear"]}
NARR = {"causality_reasoning": "No causal claim is established.", "validation_reasoning": "See the listed tests."}


def make(tmp_path, reasoner=None, fred=None, sync=True):
    app = Flask(__name__)
    init_research(app, str(tmp_path / "t.db"), sync=sync,
                  reasoner=reasoner or Reasoner(api_key=""),
                  data=DataService(fred=fred or StubFred(), alpaca=AlpacaProvider(key="", secret=""),
                                   cache_path=str(tmp_path / "t.db")))
    return app.test_client()


def run(client, body):
    r = client.post("/api/research/runs", json=body)
    assert r.status_code == 202, r.get_json()
    j = r.get_json()
    assert j["tracking_url"].endswith(j["run_id"]) and j["status"] == "queued"
    return client.get(j["tracking_url"]).get_json()


def gem(tmp_path, **kw):
    return make(tmp_path, Reasoner(client=FakeGemini(**kw)))


# ------------------------------------------------------------------ demo scenarios (planted truths)
@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return run(make(tmp_path_factory.mktemp("demo")), {"mode": "demo_autonomous"})


def test_demo_labelled_synthetic_and_offline(demo):
    assert demo["status"] == "completed" and demo["mode"] == "demo_autonomous"
    assert demo["reasoning_source"] == "template_demo" and demo["data_provenance"] == "synthetic"
    assert any("DEMO MODE" in w for w in demo["warnings"])
    assert all(v["synthetic"] for h in demo["hypotheses"] for v in h["provenance"]["variables"])


def test_demo_planted_signal(demo):
    h = demo["hypotheses"][0]
    assert h["correlation"]["association_verdict"] == "detected" and h["correlation"]["primary_lag"] == 1
    assert h["causality"]["causal_verdict"] == "evidence_consistent"
    assert h["validation"]["reliability_verdict"] == "supported"


def test_demo_null_stops_at_correlation(demo):
    h = demo["hypotheses"][1]
    assert h["correlation"]["association_verdict"] == "not_detected"
    assert h["status"] == "no_correlation"
    assert h["causality"] is None and h["validation"] is None and h["backtest"] is None and h["generalisation"] is None
    st = {s["id"]: s for s in h["stages"]}
    assert st["correlation"]["status"] == "completed"
    assert all(st[k]["status"] == "skipped" and "no correlation" in st[k]["message"] for k in
               ("causality", "validation", "backtest", "generalisation", "critique"))
    assert h["verdict"]["statement"].startswith("Initial hypothesis with no correlation")
    assert demo["summary"]["no_correlation_label"] == "Initial hypotheses with no correlation"
    assert demo["summary"]["no_correlation"] == ["H2"] and demo["summary"]["counts"]["no_correlation"] == 1
    assert "Initial hypotheses with no correlation".lower() in demo["summary"]["overall"]
    vd = demo["visualization_data"]["H2"]
    assert vd["scatter"] and vd["rolling_r"]                       # the correlation graph is still available


def test_demo_common_factor_confounding(demo):
    h = demo["hypotheses"][2]
    assert h["correlation"]["association_verdict"] == "detected"
    cc = next(t for t in h["causality"]["tests"] if t["id"] == "confounder_control")
    assert cc["result"] == "fail" and cc["metrics"]["attenuation"] >= 0.5
    assert h["causality"]["causal_verdict"] == "contradicted"


def test_every_hypothesis_has_full_structure(demo):
    for h in [x for x in demo["hypotheses"] if x["status"] == "completed"]:
        st = {s["id"]: s["status"] for s in h["stages"]}
        assert [st[k] for k in ("correlation", "causality", "validation", "backtest", "critique")] == ["completed"] * 5
        assert st["generalisation"] in ("completed", "skipped")
        assert h["verdict"]["evidence_level"] == "exploratory"
        ids = {t["id"] for t in h["causality"]["tests"]}
        assert {"negative_control", "event_study"} <= ids       # reported as not_run, never fabricated
        assert all(t["result"] == "not_run" for t in h["causality"]["tests"] if t["id"] in ("negative_control", "event_study"))   # synthetic demo: no real control series or Fed calendar
        assert h["causality"]["causal_verdict"] != "proven"
        assert h["causality"]["tests"][0]["result"] in ("pass", "fail", "inconclusive", "not_run")
        vd = demo["visualization_data"][h["hypothesis_id"]]
        assert {"timeseries", "scatter", "rolling_r", "causal_graph", "validation_metrics"} <= set(vd)
        assert any(r["r"] is None for r in vd["rolling_r"]) and vd["rolling_r"][-1]["r"] is not None
    json.dumps(demo, allow_nan=False)


def test_preregistration_frozen_before_results(demo):
    pr = demo["preregistration"]
    assert pr["frozen_before_any_statistic"] and len(pr["sha256"]) == 64 and pr["family_size"] == 3
    assert pr["frozen_at"] <= demo["timestamps"]["finished"]
    assert pr["hypotheses"][0]["plan"]["split"]["holdout"][0] > pr["hypotheses"][0]["plan"]["split"]["development"][1]


def test_multiple_testing_family_is_holm_over_all_tested(demo):
    mt = demo["multiple_testing"]
    p = mt["primary_p_hac"]
    adj, _ = S.holm([p["H1"], p["H2"], p["H3"]])
    assert [mt["primary_p_holm"][k] for k in ("H1", "H2", "H3")] == pytest.approx(adj)
    assert mt["proposed"] == 3 and mt["attempted"] == 3 and mt["correction"] == "holm"
    for h in [x for x in demo["hypotheses"] if x["validation"]]:
        fam = next(t for t in h["validation"]["tests"] if t["id"] == "family_multiple_testing")
        assert fam["critical"] and fam["metrics"]["family_size"] == 3     # the stopped null still counts in the family
    assert demo["summary"]["counts"] == {"supported": 2, "rejected": 0, "inconclusive": 0, "failed": 0, "no_correlation": 1}


def test_demo_mode_never_calls_gemini(tmp_path):
    fake = FakeGemini(boom="must not be called")
    c = make(tmp_path, Reasoner(client=fake))
    assert run(c, {"mode": "demo_autonomous"})["status"] == "completed" and fake.prompts == []


# ------------------------------------------------------------------ transforms / alignment / look-ahead
def test_transforms_numerically_correct():
    lv = pd.Series([100.0, 110.0, 99.0, 105.0, 120.0, 118.0, 130.0, 125.0],
                   index=pd.bdate_range("2024-01-01", periods=8))
    assert T.apply(lv, "log_return").iloc[1] == pytest.approx(100 * np.log(1.1))
    assert T.apply(lv, "diff").iloc[2] == pytest.approx(-11.0)
    r = 100 * np.log(lv).diff()
    fwd = T.apply(lv, "fwd_logvol_5")
    assert fwd.iloc[0] == pytest.approx(np.log(r.iloc[1:6].std(ddof=1) * np.sqrt(252)))   # uses t+1..t+5 only
    assert fwd.iloc[-5:].isna().all()
    trail = T.apply(lv, "trailing_logvol_5")
    assert trail.iloc[5] == pytest.approx(np.log(r.iloc[1:6].std(ddof=1) * np.sqrt(252)))  # uses t-4..t only


def test_no_lookahead_in_predictor_transforms():
    lv = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(size=60) / 100)) * 100)
    for tr in ("log_return", "diff", "trailing_logvol_5"):
        a = T.apply(lv, tr)
        lv2 = lv.copy(); lv2.iloc[40:] *= 3                    # alter the FUTURE
        b = T.apply(lv2, tr)
        pd.testing.assert_series_equal(a.iloc[:40], b.iloc[:40])


def test_levels_aligned_before_transform_on_mismatched_calendars():
    a = pd.Series(np.linspace(100, 200, 300), index=pd.bdate_range("2023-01-02", periods=300))
    b = a.copy() * 2 + np.arange(300) % 7
    b = b.drop(b.index[[10, 11, 50]])                           # series B has extra holidays
    h = {"predictor": {"variable_id": "DGS10", "transform": "diff"}, "outcome": {"variable_id": "SP500", "transform": "log_return"}}
    df, info = H.build_frame(h, {"DGS10": a, "SP500": b}, {})
    assert info["raw_rows_aligned"] == 297 and set(df.index) <= set(b.index)
    # x is the change over the COMMON grid (spans the dropped holiday), not over a's own grid
    assert a.index[10] not in df.index and a.index[11] not in df.index
    assert df.loc[a.index[12], "x"] == pytest.approx(a[a.index[12]] - a[a.index[9]])   # change since last COMMON date


def test_monthly_macro_unsupported_for_prediction():
    reg = R.Registry()
    v, why = reg.resolve("CPIAUCSL")
    assert why and "release-date" in why
    p = HypothesisProposal(**hyp(predictor_variable="CPIAUCSL", predictor_transform="diff"))
    norm, reasons, _ = validate_proposal(p, reg, None, False)
    assert norm is None and "CPIAUCSL" in reasons[0]


def test_stage1_and_2_never_see_holdout():
    rng = np.random.default_rng(3)
    n = 800
    idx = pd.bdate_range("2021-01-04", periods=n)
    x = rng.normal(size=n)
    y = np.r_[0, 0.3 * x[:-1]] + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "c_SPY": rng.normal(size=n)}, index=idx)
    h = {"lag_days": 1, "direction": "positive", "predictor": {"label": "x", "transform": "log_return", "variable_id": "A"},
         "outcome": {"label": "y", "transform": "log_return", "variable_id": "B"}, "control_variables": [], "market_control": ["SPY"]}
    df2 = df.copy()
    a, b = V.split_indices(n)
    df2.iloc[b:, 0] = rng.normal(size=n - b) * 50               # destroy the holdout
    df2.iloc[b:, 1] = rng.normal(size=n - b) * 50
    dev1, _ = H.split_frame(df); dev2, _ = H.split_frame(df2)
    c1, c2 = H.stage1(dev1, h), H.stage1(dev2, h)
    assert c1["r"] == c2["r"] and c1["p_value_hac"] == c2["p_value_hac"]
    prop = {"mechanisms": [], "alternative_explanations": [], "confounders": [], "assumptions": []}
    s1, s2 = H.stage2(dev1, h, prop, c1), H.stage2(dev2, h, prop, c2)
    assert [t["metrics"] for t in s1["tests"]] == [t["metrics"] for t in s2["tests"]]


def test_holdout_evaluated_once_and_isolated():
    rng = np.random.default_rng(5)
    n = 800
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": np.r_[0, 0.3 * x[:-1]] + rng.normal(size=n)}, index=pd.bdate_range("2021-01-04", periods=n))
    df2 = df.copy()
    _, b = V.split_indices(n)
    df2.iloc[b:, 1] = rng.normal(size=n - b) * 10
    v1 = V.run_validation(df, fixed_lag=1, lag_family=[1, 2, 3], development_only=True)
    v2 = V.run_validation(df2, fixed_lag=1, lag_family=[1, 2, 3], development_only=True)
    assert v1["holdout_evaluations"] == 1
    same = {t["id"] for t, u in zip(v1["tests"], v2["tests"]) if t["metrics"] == u["metrics"]}
    assert {"validation_replication", "hac_holm", "block_bootstrap", "placebo_max_stat", "window_stability",
            "outlier_sensitivity"} <= same                      # every non-holdout diagnostic ignores the holdout
    assert "holdout_single_evaluation" not in same
    assert V.run_validation(df, fixed_lag=1, development_only=True)["selected_lag"] == 1   # pre-registered, not selected


# ------------------------------------------------------------------ Gemini live-mode behaviour (fake client)
def test_autonomous_with_fake_gemini_mixed_macro_and_equity(tmp_path):
    fake = FakeGemini(generation=gen(hyp(), hyp(question="Does the VIX level predict future 5-day S&P 500 volatility?",
                                              predictor_variable="VIXCLS", predictor_transform="level",
                                              outcome_transform="fwd_logvol_5", direction="positive", lag_days=3,
                                              control_variables=[])),
                      causal=CAUSAL, narrative=NARR)
    rep = run(make(tmp_path, Reasoner(client=fake)), {"mode": "autonomous", "theme": "rates and vol", "lookback_years": 3, "max_hypotheses": 3})
    assert rep["status"] == "completed" and rep["reasoning_source"] == "gemini"
    assert rep["data_provenance"] == "real" and rep["generation"]["source"] == "gemini"
    h1, h2 = rep["hypotheses"]
    assert h1["status"] == "completed" and h1["provenance"]["variables"][0]["provider"] == "stub"
    assert h2["normalized"]["lag_days"] == 0 and any("lag_days 3 -> 0" in n for n in h2["eligibility"]["notes"])
    assert h2["status"] in ("completed", "no_correlation")
    assert "VIXCLS" in json.dumps(fake.prompts[0]) and "SYN_" not in fake.prompts[0]      # no fixtures in live catalogue
    nodes = {n["id"] for n in rep["visualization_data"]["H1"]["causal_graph"]["nodes"]}
    assert "latent:growth_news" in nodes and "EVIL;DROP" not in nodes                    # bad DAG node dropped
    assert rep["reasoning_sources"]["narrative"][0]["source"] == "gemini"
    assert h1["causality"]["reasoning"] == NARR["causality_reasoning"]


def test_unsafe_or_unsupported_proposals_rejected_with_reasons(tmp_path):
    bad = [hyp(predictor_variable="__import__('os')", question="Do dangerous identifiers get executed? Surely not."),
           hyp(predictor_variable="UNRATE", predictor_transform="diff"),
           hyp(predictor_transform="fwd_logvol_5"),
           hyp(outcome_transform="drop table"),
           hyp(proposed_test="arbitrary_python"),
           hyp(lag_days=0), hyp(lag_days=9),
           hyp(outcome_variable="DGS10", predictor_variable="DGS10"),
           hyp(predictor_variable="AAPL")]
    rep = run(gem(tmp_path, generation=gen(hyp(), *bad), causal=CAUSAL, narrative=NARR), {"mode": "autonomous", "max_hypotheses": 5})
    assert rep["hypotheses"][0]["status"] == "completed"
    for h in rep["hypotheses"][1:]:
        assert h["status"] == "rejected" and h["eligibility"]["reasons"] and h["correlation"] is None
    assert rep["multiple_testing"]["proposed"] == 10 or rep["multiple_testing"]["proposed"] == len(rep["hypotheses"])
    assert any("Alpaca" in r for r in rep["hypotheses"][-1]["eligibility"]["reasons"])


def test_over_budget_proposals_are_recorded_not_dropped(tmp_path):
    hs = [hyp(), hyp(lag_days=2), hyp(lag_days=3)]
    rep = run(gem(tmp_path, generation=gen(*hs), causal=CAUSAL, narrative=NARR), {"mode": "autonomous", "max_hypotheses": 2})
    st = [h["status"] for h in rep["hypotheses"]]
    assert st[0] == "completed" and st[1] != "rejected" and st[2] == "rejected"
    assert "ranked below the 2 most realistic" in rep["hypotheses"][2]["eligibility"]["reasons"][0]
    assert rep["multiple_testing"]["proposed"] == 3 and rep["multiple_testing"]["attempted"] == 2


def test_malformed_gemini_generation_fails_visibly_no_fake_fallback(tmp_path):
    for out in ("this is not json", {"hypotheses": "nope"}, {"unexpected": 1}):
        rep = run(gem(tmp_path, generation=out), {"mode": "autonomous"})
        assert rep["status"] == "failed" and rep["errors"][0]["code"] == "gemini_error"
        assert rep["reasoning_source"] is None and rep["hypotheses"] == []


def test_gemini_provider_outage_in_generation_fails_visibly(tmp_path, monkeypatch):
    import research.llm as L
    sleeps = []
    monkeypatch.setattr(L.time, "sleep", sleeps.append)
    rep = run(gem(tmp_path, boom="503 unavailable"), {"mode": "autonomous"})
    assert rep["status"] == "failed" and "No fallback" in rep["errors"][0]["message"]
    assert len(sleeps) == L.RETRIES - 1                      # transient 503 retried with backoff, then failed visibly


def test_transient_gemini_error_is_retried_then_succeeds(monkeypatch):
    import research.llm as L
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Flaky(FakeGemini):
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("503 UNAVAILABLE high demand")
            return type("R", (), {"text": json.dumps({"causality_reasoning": "ok", "validation_reasoning": "ok"})})()
    out = Reasoner(client=Flaky())._call("x", L.Narrative)
    assert out["causality_reasoning"] == "ok" and calls["n"] == 3


def test_guided_unsupported_question(tmp_path):
    rep = run(gem(tmp_path, generation=gen(unsupported="Cannot test astrology with this catalogue")),
              {"mode": "guided", "question": "Do planets predict the stock market?"})
    assert rep["status"] == "failed" and "astrology" in rep["errors"][0]["message"]


def test_malformed_causal_and_narrative_fall_back_labelled(tmp_path):
    bad_narr = {"causality_reasoning": "r was exactly 0.987654 and proven", "validation_reasoning": "x"}
    rep = run(gem(tmp_path, generation=gen(hyp()), causal={"nope": 1}, narrative=bad_narr),
              {"mode": "autonomous", "max_hypotheses": 1})
    h = rep["hypotheses"][0]
    assert rep["status"] == "completed" and rep["reasoning_source"] == "gemini"          # generation was genuinely Gemini
    assert rep["reasoning_sources"]["causal"][0]["source"] == "template_fallback"
    assert rep["reasoning_sources"]["narrative"][0]["source"] == "template_fallback"
    assert any("Gemini" in w for w in h["warnings"]) and any("fell back" in w for w in rep["warnings"])
    assert "0.987654" not in h["causality"]["reasoning"] and h["correlation"]["r"] is not None


def test_narrative_may_round_but_not_invent_numbers():
    from research.llm import unsupported_numbers as u
    assert u("r = 0.4749582 across 555", {"r": 0.47495821, "n": 555}) == []
    assert u("r = 0.47 and p = 0.001", {"r": 0.4749}) == ["0.001"]


# ------------------------------------------------------------------ data provider failure modes
def test_provider_failure_rejects_that_hypothesis_but_run_continues(tmp_path):
    two = gen(hyp(), hyp(predictor_variable="DGS2", question="Do 2-year yield changes predict next-day S&P 500 returns?"))
    c = make(tmp_path, Reasoner(client=FakeGemini(generation=two, causal=CAUSAL, narrative=NARR)), fred=StubFred(fail={"DGS2"}))
    rep = run(c, {"mode": "autonomous"})
    assert rep["status"] == "completed"
    assert rep["hypotheses"][1]["status"] == "rejected" and "stub failure" in rep["hypotheses"][1]["eligibility"]["reasons"][0]
    assert rep["hypotheses"][0]["status"] == "completed"
    assert rep["multiple_testing"]["attempted"] == 1


def test_all_providers_down_gives_clear_failure(tmp_path):
    c = make(tmp_path, Reasoner(client=FakeGemini(generation=gen(hyp()), causal=CAUSAL, narrative=NARR)), fred=StubFred(fail={"DGS10", "SP500"}))
    rep = run(c, {"mode": "autonomous"})
    assert rep["status"] == "failed" and rep["errors"][0]["code"] == "no_testable_hypotheses"
    assert "stub failure" in rep["errors"][0]["message"]
    assert rep["hypotheses"][0]["status"] == "rejected"          # failure preserved in history


def test_http_retry_with_backoff(monkeypatch):
    import research.providers as P
    seq = iter([429, 503, 200])
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: type("R", (), {"status_code": next(seq)})())
    assert _get("http://x").status_code == 200
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: type("R", (), {"status_code": 500})())
    with pytest.raises(ProviderError):
        _get("http://x")


def test_provider_cache(tmp_path):
    StubFred.calls = 0
    d = DataService(fred=StubFred(), alpaca=AlpacaProvider(key="", secret=""), cache_path=str(tmp_path / "c.db"))
    v = R.Registry().resolve("DGS10")[0]
    s1, p1 = d.get(v, 1)
    s2, p2 = d.get(v, 1)
    assert StubFred.calls == 1 and p1["from_cache"] is False and p2["from_cache"] is True
    assert len(s1) == len(s2) and p2["retrieved_at"] == p1["retrieved_at"]


def test_alpaca_pagination_adjustment_and_errors(monkeypatch):
    import research.providers as P
    pages = [{"bars": {"SPY": [{"t": "2024-01-02T05:00:00Z", "c": 100.0}]}, "next_page_token": "abc"},
             {"bars": {"SPY": [{"t": "2024-01-03T05:00:00Z", "c": 101.0}]}, "next_page_token": None}]
    seen = []

    def fake_get(url, **kw):
        seen.append(kw["params"])
        return type("R", (), {"status_code": 200, "json": lambda self: pages[len(seen) - 1]})()
    monkeypatch.setattr(P.requests, "get", fake_get)
    a = AlpacaProvider(key="k", secret="s")
    s, prov = a.fetch(R.Registry(equity_available=True).resolve("SPY")[0], pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-02-01").date())
    assert list(s.values) == [100.0, 101.0] and prov["adjustment"].startswith("all") and prov["feed"] == "IEX"
    assert seen[0]["adjustment"] == "all" and seen[1]["page_token"] == "abc"
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: type("R", (), {"status_code": 403})())
    with pytest.raises(ProviderError, match="rejected the credentials"):
        a.fetch(R.Registry(equity_available=True).resolve("SPY")[0], pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-02-01").date())
    with pytest.raises(ProviderError, match="not configured"):
        AlpacaProvider(key="", secret="").fetch(R.Registry().resolve("SYN_OIL")[0], None, None)


# ------------------------------------------------------------------ endpoint contract
def test_catalog_reports_real_config(tmp_path):
    c = make(tmp_path, Reasoner(client=FakeGemini()))
    cat = c.get("/api/research/catalog").get_json()
    assert cat["providers"]["gemini"]["configured"] and cat["providers"]["alpaca"]["configured"] is False
    byid = {v["id"]: v for v in cat["variables"]}
    assert byid["DGS10"]["available"] and not byid["SPY"]["available"] and "Alpaca" in byid["SPY"]["unavailable_reason"]
    assert not byid["CPIAUCSL"]["available"] and "release-date" in byid["CPIAUCSL"]["unavailable_reason"]
    assert cat["lookback_years"] == [1, 3, 5] and {m["id"] for m in cat["modes"]} == {"autonomous", "guided", "demo_autonomous"}
    assert not any(v["id"].startswith("SYN_") for v in cat["variables"])


@pytest.mark.parametrize("body,status,frag", [
    ({"mode": "nope"}, 400, "mode must be"),
    ({"mode": "autonomous", "lookback_years": 2}, 400, "lookback_years"),
    ({"mode": "autonomous", "lookback_years": True}, 400, "lookback_years"),
    ({"mode": "autonomous", "max_hypotheses": 99}, 400, "max_hypotheses"),
    ({"mode": "autonomous", "max_hypotheses": 0}, 400, "max_hypotheses"),
    ({"mode": "autonomous", "extra": 1}, 400, "unsupported fields"),
    ({"mode": "guided"}, 400, "question"),
    ({"mode": "guided", "question": "short"}, 400, "question"),
    ({"mode": "autonomous", "universe": {"tickers": ["SPY"]}}, 400, "Alpaca"),
    ({"mode": "autonomous", "universe": {"tickers": ["rm -rf /"]}}, 400, "rejected"),
    ({"mode": "autonomous", "universe": {"macro_series": ["CPIAUCSL"]}}, 400, "release-date"),
    ({"mode": "autonomous", "universe": {"macro_series": ["http://evil.example/x"]}}, 400, "rejected"),
    ({"mode": "autonomous", "universe": {"url": "x"}}, 400, "universe"),
    ({"mode": "autonomous", "universe": {"macro_series": ["DGS10"] * 9}}, 400, "max 8"),
    ({"mode": "autonomous", "theme": "x" * 300}, 400, "theme"),
    ({"mode": "demo_autonomous", "universe": {"macro_series": ["DGS10"]}}, 400, "no universe"),
    ({"preset_id": "AAPL"}, 400, "preset_id"),
])
def test_bad_requests_rejected(tmp_path, body, status, frag):
    c = make(tmp_path, Reasoner(client=FakeGemini()))
    r = c.post("/api/research/runs", json=body)
    assert r.status_code == status and frag in r.get_json()["error"], r.get_json()


def test_live_modes_need_gemini_demo_does_not(tmp_path):
    c = make(tmp_path, Reasoner(api_key=""))
    for mode in ("autonomous", "guided"):
        r = c.post("/api/research/runs", json={"mode": mode, "question": "Does oil predict airline returns?"})
        assert r.status_code == 503 and "Gemini is not configured" in r.get_json()["error"]
    assert c.post("/api/research/runs", json={"mode": "demo_autonomous"}).status_code == 202


def test_oversized_and_non_json_bodies(tmp_path):
    c = make(tmp_path, Reasoner(client=FakeGemini()))
    assert c.post("/api/research/runs", data="x" * 30000, content_type="application/json").status_code in (400, 413)
    assert c.post("/api/research/runs", data="nope").status_code == 400


def test_active_run_limit_429(tmp_path):
    c = make(tmp_path, Reasoner(client=FakeGemini()), sync=True)
    from flask import current_app  # noqa: F401
    store = c.application.extensions["research"]["store"]
    from research.engine import new_report
    for i in range(3):
        store.create(f"x{i}", "autonomous", new_report(f"x{i}", "autonomous", {}))
    r = c.post("/api/research/runs", json={"mode": "demo_autonomous"})
    assert r.status_code == 429


def test_legacy_preset_requests_still_work(tmp_path):
    c = make(tmp_path)
    r = c.post("/api/research/runs", json={"preset_id": "demo_effect"})
    assert r.status_code == 202
    rep = c.get(f"/api/research/runs/{r.get_json()['run_id']}").get_json()
    assert rep["schema_version"] == "1.0" and rep["causality"]["causal_verdict"] == "evidence_consistent"
    assert [p["id"] for p in c.get("/api/research/presets").get_json()["presets"]] == ["demo_null", "demo_confounder", "demo_effect"]


def test_run_history_includes_failures_and_modes(tmp_path):
    c = make(tmp_path, Reasoner(client=FakeGemini(generation="garbage")))
    run(c, {"mode": "autonomous"})
    run(c, {"mode": "demo_autonomous"})
    c.post("/api/research/runs", json={"preset_id": "demo_null"})
    runs = c.get("/api/research/runs?limit=10").get_json()["runs"]
    assert sorted((r["mode"], r["status"]) for r in runs) == [("autonomous", "failed"), ("demo", "completed"), ("demo_autonomous", "completed")]
    assert c.get("/api/research/runs?limit=abc").status_code == 400


def test_threaded_run_progress_phases_in_order(tmp_path):
    import time
    c = make(tmp_path, sync=False)
    rid = c.post("/api/research/runs", json={"mode": "demo_autonomous"}).get_json()["run_id"]
    phases = []
    for _ in range(600):
        rep = c.get(f"/api/research/runs/{rid}").get_json()
        phases.append(rep["current_phase"])
        if rep["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert rep["status"] == "completed"
    first = list(dict.fromkeys(phases))                          # first appearance of each phase
    order = ["queued", "hypothesis_generation", "preregistered", "correlation", "causality", "validation", "backtest", "generalisation", "family_correction", "critique", "complete"]
    # phases repeat per hypothesis and polling can miss fast ones, so assert only the invariants
    assert set(first) <= set(order) and first[0] in ("queued", "hypothesis_generation") and first[-1] == "complete"
    assert first.index("complete") == len(first) - 1


def test_timeout_marks_failed_and_keeps_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_JOB_TIMEOUT", "0")
    rep = run(make(tmp_path), {"mode": "demo_autonomous"})
    assert rep["status"] == "failed" and rep["errors"][-1]["code"] == "timeout"


def test_no_secrets_in_reports_or_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-GEMINI-123")
    monkeypatch.setenv("ALPACA_API_KEY", "SECRET-ALPACA-123")
    monkeypatch.setenv("FRED_API_KEY", "SECRET-FRED-123")
    app = Flask(__name__)
    init_research(app, str(tmp_path / "t.db"), sync=True, data=DataService(fred=StubFred(), cache_path=str(tmp_path / "t.db")))
    c = app.test_client()
    rep = run(c, {"mode": "demo_autonomous"})
    blob = json.dumps(rep) + json.dumps(c.get("/api/research/catalog").get_json()) + json.dumps(c.get("/api/health").get_json())
    assert "SECRET-" not in blob
