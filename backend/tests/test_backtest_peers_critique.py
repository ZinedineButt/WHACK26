"""Backtest, peer generalisation, literature retrieval and grounded self-critique."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import autollm, backtest as BT, peers as PEERS, registry as R, validation as V
from research.api import Services
from research.llm import Reasoner
from research.providers import AlpacaProvider, DataService
from tests.test_autonomous import CAUSAL, NARR, FakeGemini, StubFred, gen, hyp, make, run

H = {"lag_days": 1, "direction": "positive", "predictor": {"variable_id": "A", "label": "a", "transform": "log_return"},
     "outcome": {"variable_id": "B", "label": "b", "transform": "log_return"}}


def frame(n=800, beta=0.4, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.r_[0, beta * x[:-1]] + rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y}, index=pd.bdate_range("2021-01-04", periods=n))


# ------------------------------------------------------------------ backtest
def test_simulate_uses_lagged_return_and_charges_turnover_exactly():
    x = np.array([1.0, -1.0, -1.0, 1.0, 0.5])
    y = np.array([9.0, 2.0, -3.0, 4.0, 5.0])        # y[0] is never traded (needs lag 1)
    pos, yo, gross, net, cost = BT.simulate(x, y, lag=1, sign=1, cost_bps=10)
    assert list(pos) == [1, -1, -1, 1]              # positions from x[0..3]
    assert list(yo) == [2.0, -3.0, 4.0, 5.0]        # earned on the NEXT day
    assert list(gross) == [2.0, 3.0, -4.0, 5.0]
    assert list(cost) == pytest.approx([0.1, 0.2, 0.0, 0.2])        # |dpos| * 10bps, in % units
    assert list(net) == pytest.approx(gross - cost)


def test_backtest_signal_has_no_lookahead():
    df = frame()
    y2 = df["y"].to_numpy().copy()
    y2[400:] = 0.0                                    # change the FUTURE of y
    a = BT.simulate(df["x"].to_numpy(), df["y"].to_numpy(), 1, 1)[0]
    b = BT.simulate(df["x"].to_numpy(), y2, 1, 1)[0]
    assert np.array_equal(a, b)                       # positions depend on x only
    ga = BT.simulate(df["x"].to_numpy(), df["y"].to_numpy(), 1, 1)[2]
    gb = BT.simulate(df["x"].to_numpy(), y2, 1, 1)[2]
    assert np.array_equal(ga[:398], gb[:398])


def test_backtest_planted_signal_passes_and_null_does_not():
    good = BT.run_backtest(frame(beta=0.4), H, dev_r=0.3)
    assert good["applicable"] and good["verdict"] == "pass" and good["holdout"]["sharpe"] > 1
    assert good["holdout"]["mean_daily_net_ci_95"][0] > 0
    bad = BT.run_backtest(frame(beta=0.0, seed=2), H, dev_r=0.0)
    assert bad["verdict"] in ("fail", "inconclusive")


def test_backtest_costs_and_holdout_isolation():
    df = frame()
    free = BT.run_backtest(df, H, dev_r=0.3, cost_bps=0)["holdout"]["total_return_pct"]
    pricey = BT.run_backtest(df, H, dev_r=0.3, cost_bps=50)["holdout"]["total_return_pct"]
    assert pricey < free
    df2 = df.copy()
    _, b = V.split_indices(len(df))
    df2.iloc[b:, 1] = np.random.default_rng(0).normal(size=len(df) - b)
    d1, d2 = BT.run_backtest(df, H, dev_r=0.3), BT.run_backtest(df2, H, dev_r=0.3)
    assert d1["development"] == d2["development"] and d1["holdout"] != d2["holdout"] and d1["sign"] == d2["sign"]


def test_direction_any_takes_sign_from_development_only():
    assert BT.frozen_sign({**H, "direction": "any"}, -0.2) == -1
    assert BT.frozen_sign({**H, "direction": "any"}, 0.2) == 1
    assert BT.frozen_sign({**H, "direction": "negative"}, 0.9) == -1


def test_backtest_not_applicable_for_volatility_outcomes():
    h = {**H, "outcome": {**H["outcome"], "transform": "fwd_logvol_5"}}
    out = BT.run_backtest(frame(), h, dev_r=0.1)
    assert out["applicable"] is False and out["verdict"] == "not_run" and "not a P&L" in out["reason"]


def test_curve_holdout_rebased_and_json_strict():
    out = BT.run_backtest(frame(), H, dev_r=0.3)
    c = out["equity_curve"]
    first = next(r for r in c if r["sample"] == "holdout")
    assert abs(first["strategy_holdout"]) < 5
    assert all(r["strategy_holdout"] is None for r in c if r["sample"] == "development")
    json.dumps(out, allow_nan=False)


# ------------------------------------------------------------------ peers
def test_peer_candidates_fixed_map():
    reg = R.Registry(equity_available=True)
    c = PEERS.candidates("QQQ", "DGS10", reg)
    assert "QQQ" not in c and len(c) <= 5 and c[0] in ("SPY", "IWM", "DIA", "XLK", "SP500")
    assert "SP500" not in PEERS.candidates("NASDAQCOM", "SP500", R.Registry())          # predictor excluded
    no_alpaca = PEERS.candidates("QQQ", "DGS10", R.Registry(equity_available=False))
    assert all(R.Registry().resolve(x)[0].source == "fred" for x in no_alpaca)           # only available peers
    assert PEERS.candidates("DGS10", "DGS2", reg) == []                                  # rates have no price peers


def test_peer_replication_in_demo_and_pre_registered_skip_rule(tmp_path):
    rep = run(make(tmp_path), {"mode": "demo_autonomous"})
    h1, h2 = rep["hypotheses"][0], rep["hypotheses"][1]
    g = h1["generalisation"]
    by = {p["variable_id"]: p for p in g["peers"]}
    assert by["SYN_AIR2"]["significant_after_holm"] and by["SYN_AIR2"]["same_sign_as_primary"]   # planted effect replicates
    assert not by["SYN_TECH"]["significant_after_holm"]                                           # no effect there
    assert g["verdict"] == "partial" and "holm" in g["correction"]
    assert h2["status"] == "no_correlation" and h2["generalisation"] is None      # null: peers never tested
    plan = rep["preregistration"]["hypotheses"][0]["plan"]
    assert "SYN_AIR2" in plan["peer_candidates"] and plan["backtest"]["cost_bps"] == 2.0           # frozen before results
    assert h1["backtest"]["applicable"] and "equity_curve" not in h1["backtest"]
    assert rep["visualization_data"]["H1"]["backtest_equity"] and rep["visualization_data"]["H1"]["peers"]


def test_peer_fetch_failure_is_recorded_not_fatal():
    svc = Services(Reasoner(api_key=""), Reasoner(api_key=""),
                   DataService(fred=StubFred(fail={"NASDAQCOM"}), alpaca=AlpacaProvider(key="", secret="")))
    h = {**H, "predictor": {"variable_id": "DGS10", "label": "x", "transform": "diff"},
         "outcome": {"variable_id": "SP500", "label": "y", "transform": "log_return"}}
    levels = {"DGS10": svc.data.get(svc.registry.resolve("DGS10")[0], 3)}
    out = PEERS.run_peers(h, ["NASDAQCOM", "DJIA"], svc, 3, levels, 1)
    assert [s["variable_id"] for s in out["skipped"]] == ["NASDAQCOM"] and out["n_peers"] == 1
    assert PEERS.run_peers(h, ["NASDAQCOM"], svc, 3, levels, 1)["verdict"] == "not_run"


# ------------------------------------------------------------------ literature
class _Resp:
    status_code = 200

    def json(self):
        return {"results": [
            {"title": "Yields and Stock Returns", "publication_year": 2015, "doi": "https://doi.org/10.1/abc", "cited_by_count": 10,
             "abstract_inverted_index": {"Yields": [0], "predict": [1], "returns": [2]},
             "primary_location": {"source": {"display_name": "JF"}}},
            {"title": "", "abstract_inverted_index": None}]}


def test_literature_parses_real_shape_and_sanitises_query(monkeypatch):
    import research.literature as L
    seen = {}
    monkeypatch.setattr(L, "_get", lambda url, **kw: (seen.update(kw["params"]) or _Resp()))
    out = L.search({"predictor": {"label": "10y <script>alert(1)</script> yield"}, "outcome": {"label": "S&P 500"}})
    assert out["status"] == "completed" and out["papers"][0]["abstract"] == "Yields predict returns"
    assert out["papers"][0]["index"] == 1 and len(out["papers"]) == 1
    assert "<" not in seen["search"] and ">" not in seen["search"]


def test_literature_failure_degrades_gracefully(monkeypatch):
    import research.literature as L
    from research.providers import ProviderError

    def boom(url, **kw):
        raise ProviderError("down")
    monkeypatch.setattr(L, "_get", boom)
    out = L.search({"predictor": {"label": "a"}, "outcome": {"label": "b"}})
    assert out["status"] == "unavailable" and out["papers"] == []


# ------------------------------------------------------------------ self-critique grounding
FACTS = {"association_verdict": "detected", "causal_verdict": "not_established", "reliability_verdict": "supported",
         "backtest_verdict": "fail", "generalisation_verdict": "partial", "r": 0.2, "n": 500,
         "failed_tests": ["window_stability"], "all_tests": ["window_stability", "granger", "holdout_single_evaluation"]}
PAPERS = [{"index": 1, "title": "T1", "year": 2010, "abstract": "x"}, {"index": 2, "title": "T2", "year": 2012, "abstract": "y"}]


def crit(**kw):
    base = {"paper_assessments": [{"paper_index": 1, "relevance": "high", "stance": "contradicts", "note": "n"}],
            "critique": [{"point": "Window stability failed.", "severity": "high", "grounded_in": "test:window_stability"}],
            "final_assessment": "Weak.", "confidence": "high"}
    base.update(kw)
    return base


def test_critique_confidence_is_capped_by_evidence():
    out, src, _ = autollm.self_critique(Reasoner(client=FakeGemini(narrative=crit())), FACTS, PAPERS)
    assert src == "gemini" and out["confidence"] == "moderate" and out["confidence_cap"] == "moderate"
    out2, _, _ = autollm.self_critique(Reasoner(client=FakeGemini(narrative=crit())), {**FACTS, "association_verdict": "not_detected"}, PAPERS)
    assert out2["confidence"] == "low"


def test_critique_drops_ungrounded_points_and_invented_papers():
    c = crit(critique=[{"point": "Fake.", "severity": "high", "grounded_in": "test:does_not_exist"},
                       {"point": "Cites a missing paper.", "severity": "low", "grounded_in": "paper:9"},
                       {"point": "Real.", "severity": "medium", "grounded_in": "paper:2"}],
             paper_assessments=[{"paper_index": 1, "relevance": "low", "stance": "neutral", "note": "ok"},
                                {"paper_index": 7, "relevance": "high", "stance": "supports", "note": "invented"}])
    out, _, _ = autollm.self_critique(Reasoner(client=FakeGemini(narrative=c)), FACTS, PAPERS)
    assert [p["grounded_in"] for p in out["critique"]] == ["paper:2"]
    assert [a["paper_index"] for a in out["paper_assessments"]] == [1]


@pytest.mark.parametrize("bad", [crit(final_assessment="This is proven with r = 0.987."),
                                 crit(critique=[{"point": "No refs", "severity": "low", "grounded_in": "vibes"}]),
                                 {"nope": 1}])
def test_critique_bad_output_falls_back_labelled(bad):
    out, src, w = autollm.self_critique(Reasoner(client=FakeGemini(narrative=bad)), FACTS, PAPERS)
    assert src == "template_fallback" and out["source"] == "template_fallback" and w
    assert any(c["grounded_in"] == "test:window_stability" for c in out["critique"])


# ------------------------------------------------------------------ end to end (fake Gemini, stub FRED)
def test_full_live_flow_has_backtest_peers_literature_and_critique(tmp_path, monkeypatch):
    import research.engine as E
    paper = {"index": 1, "title": "A paper", "year": 2020, "abstract": "abs", "doi": "https://doi.org/x",
             "url": "https://doi.org/x", "venue": "V", "cited_by": 3, "source": "openalex"}
    monkeypatch.setattr(E.LIT, "search", lambda h, **kw: {"status": "completed", "query": "q", "provider": "openalex", "note": "", "papers": [paper]})
    c = crit(critique=[{"point": "Data snooping risk.", "severity": "medium", "grounded_in": "paper:1"}])
    fake = FakeGemini(generation=gen(hyp(outcome_variable="SP500")), causal=CAUSAL, narrative=NARR)
    orig = fake.generate_content

    def router(model, contents, config):
        fake.narrative = c if "hostile peer reviewer" in contents else NARR
        return orig(model, contents, config)
    fake.generate_content = router
    rep = run(make(tmp_path, Reasoner(client=fake)), {"mode": "autonomous", "max_hypotheses": 1})
    h = rep["hypotheses"][0]
    assert rep["status"] == "completed" and h["critique"]["source"] == "gemini"
    assert h["literature"]["papers"][0]["title"] == "A paper" and h["critique"]["critique"][0]["grounded_in"] == "paper:1"
    assert h["verdict"]["confidence"] in ("low", "moderate") and "backtest" in h["verdict"]
    assert {s["id"] for s in h["stages"]} == {"correlation", "causality", "validation", "backtest", "generalisation", "critique"}
    assert rep["reasoning_sources"]["critique"][0]["source"] == "gemini"
    json.dumps(rep, allow_nan=False)


# ------------------------------------------------------------------ literature source fallback chain
ARXIV_XML = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2001.00001v1</id><published>2020-01-02T00:00:00Z</published><title>Oil and
 Airline Returns</title><summary>We study   oil shocks and airline stocks.</summary></entry></feed>"""


class _X:
    def __init__(self, code, text="", js=None):
        self.status_code, self.text, self._js = code, text, js

    def json(self):
        return self._js


def test_literature_falls_back_openalex_to_semantic_to_arxiv(monkeypatch):
    import research.literature as L
    calls = []

    def fake(url, **kw):
        calls.append(url.split("/")[2])
        if "openalex" in url:
            return _X(429)
        if "semanticscholar" in url:
            return _X(429)
        return _X(200, ARXIV_XML)
    monkeypatch.setattr(L, "_get", fake)
    out = L.search(None, query="oil price airline stocks")
    assert calls == ["api.openalex.org", "api.semanticscholar.org", "export.arxiv.org"]
    assert out["status"] == "completed" and out["provider"] == "arxiv" and "openalex" in out["note"]
    p = out["papers"][0]
    assert p["title"] == "Oil and Airline Returns" and p["year"] == 2020 and p["abstract"] == "We study oil shocks and airline stocks."
    assert p["url"] == "http://arxiv.org/abs/2001.00001v1" and p["index"] == 1


def test_literature_uses_first_source_that_returns_papers_and_reports_total_failure(monkeypatch):
    import research.literature as L
    sem = {"data": [{"title": "Sem paper", "year": 2019, "abstract": "abs", "externalIds": {"DOI": "10.1/z"}, "url": "u", "venue": "V", "citationCount": 2}]}
    monkeypatch.setattr(L, "_get", lambda url, **kw: _X(429) if "openalex" in url else _X(200, js=sem))
    out = L.search(None, query="x y z")
    assert out["provider"] == "semantic_scholar" and out["papers"][0]["doi"] == "https://doi.org/10.1/z"
    monkeypatch.setattr(L, "_get", lambda url, **kw: _X(429))
    out = L.search(None, query="x y z")
    assert out["status"] == "unavailable" and out["papers"] == [] and "openalex" in out["note"]


def test_openalex_key_is_sent_when_configured(monkeypatch):
    import research.literature as L
    seen = {}
    monkeypatch.setenv("OPENALEX_API_KEY", "oa-key")
    monkeypatch.setattr(L, "_get", lambda url, **kw: (seen.update(kw["params"]) or _X(200, js={"results": []})))
    L._openalex("q", 3)
    assert seen["api_key"] == "oa-key"
