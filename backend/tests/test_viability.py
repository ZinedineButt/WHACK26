"""Literature-grounded viability, candidate screening, stop-at-no-correlation, Alpaca credential check."""
import json

import pytest

import research.engine as E
from research import autollm
from research.llm import Reasoner
from research.providers import AlpacaProvider, DataService
from research.api import Services
from tests.test_autonomous import CAUSAL, NARR, FakeGemini, StubFred, gen, hyp, make, run

PAPER = lambda i, t="A paper": {"index": i, "title": f"{t} {i}", "year": 2015, "abstract": "abstract text", "doi": None,
                                "url": f"https://doi.org/x{i}", "venue": "J", "cited_by": 5, "source": "openalex"}


def lit_ok(monkeypatch, n=2):
    monkeypatch.setattr(E.LIT, "search", lambda h, **kw: {"status": "completed", "query": "q", "provider": "openalex", "note": "",
                                                          "papers": [PAPER(i) for i in range(1, n + 1)]})


def rank(*items):
    return {"assessments": [{"candidate_index": i, "viability": v, "expectation": e, "summary": s, "cited_papers": c, "concerns": ""}
                            for i, v, e, s, c in items]}


def H(lag, q):
    return hyp(lag_days=lag, question=q)


# ------------------------------------------------------------------ over-generate, screen data first, rank by literature
def test_candidates_failing_data_screening_never_consume_the_budget(tmp_path, monkeypatch):
    lit_ok(monkeypatch)
    cands = gen(hyp(predictor_variable="DGS2", question="Do 2-year yield changes predict next-day S&P 500 returns?"),
                hyp(predictor_variable="T10Y2Y", question="Does the yield-curve slope change predict next-day S&P 500 returns?"),
                hyp(question="Do 10-year yield changes predict next-day S&P 500 returns?"),
                hyp(predictor_variable="DFF", question="Do fed funds changes predict next-day S&P 500 returns?"))
    fake = FakeGemini(generation=cands, causal=CAUSAL, narrative=NARR)
    c = make(tmp_path, Reasoner(client=fake), fred=StubFred(fail={"DGS2", "T10Y2Y"}))
    rep = run(c, {"mode": "autonomous", "max_hypotheses": 2})
    st = {h["hypothesis_id"]: h for h in rep["hypotheses"]}
    assert st["H1"]["status"] == st["H2"]["status"] == "rejected" and "data screening failed" in st["H1"]["eligibility"]["reasons"][0]
    assert {st["H3"]["status"], st["H4"]["status"]} <= {"completed", "no_correlation"}       # both viable ones were tested
    assert rep["multiple_testing"]["attempted"] == 2 and rep["generation"]["candidate_pool"] == 4
    assert rep["multiple_testing"]["candidates_with_data"] == 2


def test_literature_viability_selects_most_realistic_and_records_summary(tmp_path, monkeypatch):
    lit_ok(monkeypatch)
    cands = gen(H(1, "Do 10-year yield changes predict next-day S&P 500 returns today?"),
                hyp(predictor_variable="DFF", question="Do fed funds changes predict next-day S&P 500 returns reliably?"),
                hyp(predictor_variable="DGS2", question="Do 2-year yield changes predict next-day S&P 500 returns at all?"))
    ranking = rank((1, 2, "mixed", "Related work is mixed on short-horizon yield effects.", [1]),
                   (2, 5, "supports", "Policy-rate surprises are well documented to move equities.", [1, 2]),
                   (3, 1, "contradicts", "Two-year changes are largely arbitraged away in the literature.", [2]))
    fake = FakeGemini(generation=cands, causal=CAUSAL, narrative=NARR, ranking=ranking)
    rep = run(make(tmp_path, Reasoner(client=fake)), {"mode": "autonomous", "max_hypotheses": 2})
    st = {h["hypothesis_id"]: h for h in rep["hypotheses"]}
    assert st["H2"]["status"] != "rejected" and st["H1"]["status"] != "rejected"        # both fit the budget of 2
    assert st["H3"]["status"] == "rejected" and "unrealistic" in st["H3"]["eligibility"]["reasons"][0]
    lp = st["H2"]["literature_prior"]
    assert lp["viability"] == 5 and lp["expectation"] == "supports" and lp["cited_papers"] == [1, 2] and lp["source"] == "gemini"
    assert "documented" in lp["summary"] and lp["papers_retrieved"] == 2
    assert rep["reasoning_sources"]["ranking"] == "gemini"
    assert {x["hypothesis_id"]: x["plan"]["viability_score"] for x in rep["preregistration"]["hypotheses"]} == {"H1": 2, "H2": 5}
    assert "viability" in rep["preregistration"]["selection"] or "viability" in rep["preregistration"]["selection"].lower()


def test_generation_prompt_contains_background_literature(tmp_path, monkeypatch):
    lit_ok(monkeypatch, n=3)
    fake = FakeGemini(generation=gen(hyp()), causal=CAUSAL, narrative=NARR)
    run(make(tmp_path, Reasoner(client=fake)), {"mode": "autonomous", "theme": "rates and equities", "max_hypotheses": 1})
    gp = next(p for p in fake.prompts if "autonomous quantitative research agent" in p)
    assert "BACKGROUND LITERATURE" in gp and "viable and realistic" in gp and "A paper 1" in gp
    assert "Propose 2 distinct" in gp                                                   # over-generates 2x the budget


def test_rank_validation_ignores_bad_indices_citations_and_proof_claims():
    cands = [{"index": 1, "question": "q", "predictor": "p", "outcome": "o", "mechanism": "m", "papers": [PAPER(1), PAPER(2)]},
             {"index": 2, "question": "q2", "predictor": "p", "outcome": "o", "mechanism": "m", "papers": []}]
    bad = rank((1, 9, "supports", "Fine summary.", [1, 7]),          # viability clamped to 5, paper 7 invented -> dropped
               (2, 4, "supports", "This is proven to work.", []),   # claims proof -> ignored (template used)
               (99, 5, "supports", "Unknown candidate.", []))
    out, src, _ = autollm.rank_candidates(Reasoner(client=FakeGemini(ranking=bad)), cands)
    assert src == "gemini" and out[1]["viability"] == 5 and out[1]["cited_papers"] == [1]
    assert out[2]["source"] == "gemini" or out[2]["expectation"] == "insufficient"
    assert out[2]["summary"].startswith("Literature was not assessed")
    out2, src2, w = autollm.rank_candidates(Reasoner(client=FakeGemini(ranking={"nope": 1})), cands)
    assert src2 == "template_fallback" and w and out2[1]["viability"] == 3


def test_ranking_failure_keeps_proposed_order_and_labels_it(tmp_path, monkeypatch):
    lit_ok(monkeypatch)
    fake = FakeGemini(generation=gen(hyp(), hyp(lag_days=2), hyp(lag_days=3)), causal=CAUSAL, narrative=NARR, ranking="garbage")
    rep = run(make(tmp_path, Reasoner(client=fake)), {"mode": "autonomous", "max_hypotheses": 1})
    assert rep["status"] == "completed" and rep["reasoning_sources"]["ranking"] == "template_fallback"
    assert rep["hypotheses"][0]["status"] != "rejected" and rep["hypotheses"][1]["status"] == "rejected"
    assert any("ranking failed" in w for w in rep["warnings"])
    assert "not assessed" in rep["hypotheses"][0]["literature_prior"]["summary"]


# ------------------------------------------------------------------ stop at no correlation
def test_live_null_hypothesis_stops_after_correlation(tmp_path, monkeypatch):
    lit_ok(monkeypatch)
    fake = FakeGemini(generation=gen(hyp(predictor_variable="DGS2", predictor_transform="diff", outcome_variable="NASDAQCOM",
                                         question="Do 2-year yield changes predict next-day NASDAQ returns?")),
                      causal=CAUSAL, narrative=NARR)
    rep = run(make(tmp_path, Reasoner(client=fake)), {"mode": "autonomous", "max_hypotheses": 1})
    h = rep["hypotheses"][0]
    assert h["correlation"]["association_verdict"] == "not_detected" and h["status"] == "no_correlation"
    assert h["causality"] is None and h["validation"] is None and h["backtest"] is None
    assert not any("causal" in x["source"] for x in rep["reasoning_sources"]["causal"]) and rep["reasoning_sources"]["causal"] == []
    assert not any("hostile peer reviewer" in p or "sceptical econometrician" in p for p in fake.prompts)   # no wasted AI calls
    assert rep["summary"]["no_correlation_label"] == "Initial hypotheses with no correlation"
    assert h["literature_prior"] is not None                                           # related-research summary still available
    assert rep["status"] == "completed"
    json.dumps(rep, allow_nan=False)


def test_detected_hypothesis_still_runs_every_stage(tmp_path, monkeypatch):
    lit_ok(monkeypatch)
    rep = run(make(tmp_path, Reasoner(client=FakeGemini(generation=gen(hyp()), causal=CAUSAL, narrative=NARR))),
              {"mode": "autonomous", "max_hypotheses": 1})
    h = rep["hypotheses"][0]
    assert h["correlation"]["association_verdict"] == "detected" and h["status"] == "completed"
    assert h["causality"] and h["validation"] and h["backtest"] and h["critique"]


# ------------------------------------------------------------------ Alpaca credential check
class _R:
    def __init__(self, code):
        self.status_code = code


def test_alpaca_rejected_credentials_are_not_usable(monkeypatch):
    import research.providers as P
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: _R(401))
    a = AlpacaProvider(key="k", secret="s")
    ok, why = a.check()
    assert ok is False and "rejected the credentials" in why and a.usable is False
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: (_ for _ in ()).throw(AssertionError("must be cached")))
    assert a.check() == (False, why)                                                    # cached, no second call


def test_alpaca_good_and_unconfigured_and_network_error(monkeypatch):
    import research.providers as P
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: _R(200))
    assert AlpacaProvider(key="k", secret="s").check() == (True, "ok")
    assert AlpacaProvider(key="", secret="").check()[0] is False
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: (_ for _ in ()).throw(P.requests.ConnectionError()))
    assert AlpacaProvider(key="k", secret="s").check()[0] is True                       # network != bad keys


def test_rejected_alpaca_keys_hide_tickers_and_give_actionable_errors(tmp_path, monkeypatch):
    import research.providers as P
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: _R(401))
    from flask import Flask
    from research.api import init_research
    app = Flask(__name__)
    fake = FakeGemini(generation=gen(hyp()), causal=CAUSAL, narrative=NARR)
    init_research(app, str(tmp_path / "t.db"), sync=True, reasoner=Reasoner(client=fake),
                  data=DataService(fred=StubFred(), alpaca=AlpacaProvider(key="k", secret="s"), cache_path=str(tmp_path / "t.db")))
    c = app.test_client()
    cat = c.get("/api/research/catalog").get_json()
    assert cat["providers"]["alpaca"]["configured"] is True and cat["providers"]["alpaca"]["usable"] is False
    assert "rejected the credentials" in cat["providers"]["alpaca"]["status"]
    byid = {v["id"]: v for v in cat["variables"]}
    assert byid["SPY"]["available"] is False and "rejected the credentials" in byid["SPY"]["unavailable_reason"]
    r = c.post("/api/research/runs", json={"mode": "autonomous", "universe": {"tickers": ["SPY"]}})
    assert r.status_code == 400 and "rejected the credentials" in r.get_json()["error"]
    assert c.get("/api/research/health").get_json()["alpaca_usable"] is False
    run_ = c.post("/api/research/runs", json={"mode": "autonomous"})                    # still works with FRED-only catalogue
    assert run_.status_code == 202
    rep = c.get(run_.get_json()["tracking_url"]).get_json()
    assert rep["status"] == "completed" and not any(v.isupper() and len(v) <= 4 and v in ("SPY", "QQQ") for v in rep["generation"]["catalog"])


# ------------------------------------------------------------------ topic relevance / proxies / ordering
def test_prompt_demands_topic_relevance_and_proxies_and_does_not_default_to_rates(tmp_path, monkeypatch):
    lit_ok(monkeypatch)
    fake = FakeGemini(generation=gen(hyp()), causal=CAUSAL, narrative=NARR)
    run(make(tmp_path, Reasoner(client=fake)), {"mode": "guided", "question": "Is there a relationship between demand for AI and NVIDIA?"})
    gp = next(p for p in fake.prompts if "autonomous quantitative research agent" in p)
    assert "TOPIC FIRST" in gp and "Do NOT default to Treasury yields" in gp and "PROXIES" in gp and "closest measurable PROXIES" in gp
    assert "stock/ETF tickers are not available" in gp                                  # honest when Alpaca is unusable
    cat = gp[gp.index("VARIABLE CATALOGUE"):]
    assert cat.index("SP500") < cat.index("DGS10") and cat.index("VIXCLS") < cat.index("DGS10")   # rates are listed last


def test_any_valid_ticker_allowed_when_alpaca_usable():
    from research import registry as R
    p = autollm.generation_prompt("guided", R.Registry(equity_available=True).live_catalog(), 2, 3, "", "AI demand and PLTR", "", None, tickers_ok=True)
    assert "ANY valid US stock or ETF ticker" in p and "PLTR" in p and "SMH" in p
    q = autollm.generation_prompt("guided", R.Registry().live_catalog(), 2, 3, "", "x", "", None, tickers_ok=False)
    assert "ANY valid US stock" not in q


def test_unsupported_question_failure_explains_the_data_blocker(tmp_path):
    fake = FakeGemini(generation=gen(unsupported="There is no proxy for AI demand."))
    rep = run(make(tmp_path, Reasoner(client=fake)), {"mode": "guided", "question": "Is there a relationship between demand for AI and NVIDIA?"})
    msg = rep["errors"][0]["message"]
    assert rep["status"] == "failed" and "no proxy for AI demand" in msg
    assert "stock/ETF data is unavailable" in msg and "TIINGO_API_KEY" in msg


# ------------------------------------------------------------------ Tiingo / provider-agnostic stock data
class _TR:
    def __init__(self, code, body=None):
        self.status_code, self._b = code, body

    def json(self):
        return self._b


def test_tiingo_parses_adjusted_close_and_maps_errors(monkeypatch):
    import research.providers as P
    from research import registry as R
    from research.providers import ProviderError, TiingoProvider
    rows = [{"date": "2024-01-02T00:00:00.000Z", "close": 100, "adjClose": 98.5}, {"date": "2024-01-03T00:00:00.000Z", "close": 101, "adjClose": 99.4}]
    seen = {}
    monkeypatch.setattr(P, "_get", lambda url, **kw: (seen.update(url=url, **kw) or _TR(200, rows)))
    v = R.Registry(equity_available=True).resolve("NVDA")[0]
    s, prov = TiingoProvider(key="k").fetch(v, __import__("datetime").date(2024, 1, 1), __import__("datetime").date(2024, 2, 1))
    assert list(s.values) == [98.5, 99.4] and prov["provider"] == "tiingo" and "dividends" in prov["adjustment"]
    assert seen["url"].endswith("/NVDA/prices") and seen["headers"]["Authorization"] == "Token k"
    for code, frag in ((401, "rejected the key"), (404, "does not recognise symbol NVDA"), (500, "HTTP 500")):
        monkeypatch.setattr(P, "_get", lambda url, c=code, **kw: _TR(c, []))
        with pytest.raises(ProviderError, match=frag):
            TiingoProvider(key="k").fetch(v, None, None) if False else TiingoProvider(key="k").fetch(v, __import__("datetime").date(2024, 1, 1), __import__("datetime").date(2024, 2, 1))
    monkeypatch.setattr(P, "_get", lambda url, **kw: _TR(200, []))
    with pytest.raises(ProviderError, match="no prices"):
        TiingoProvider(key="k").fetch(v, __import__("datetime").date(2024, 1, 1), __import__("datetime").date(2024, 2, 1))
    with pytest.raises(ProviderError, match="not configured"):
        TiingoProvider(key="").fetch(v, None, None)


def test_equity_provider_routing_prefers_tiingo_then_alpaca_then_explains(monkeypatch):
    import research.providers as P
    from research.providers import TiingoProvider
    codes = {"tiingo": 200, "alpaca": 200}

    def fake_get(url, **kw):
        return _R(codes["tiingo"] if "tiingo" in url else codes["alpaca"])
    monkeypatch.setattr(P.requests, "get", fake_get)
    mk = lambda tk, ak: DataService(fred=StubFred(), tiingo=TiingoProvider(key=tk), alpaca=AlpacaProvider(key=ak, secret=ak))
    ok, why, prov = mk("t", "a").equity_status()
    assert ok and prov.name == "tiingo"                                                  # Tiingo wins
    codes["tiingo"] = 401
    ok, why, prov = mk("t", "a").equity_status()
    assert ok and prov.name == "alpaca"                                                  # falls back to Alpaca
    codes["alpaca"] = 401
    ok, why, prov = mk("t", "a").equity_status()
    assert not ok and prov is None and "Tiingo rejected" in why and "Alpaca rejected" in why
    ok, why, prov = mk("", "").equity_status()
    assert not ok and "TIINGO_API_KEY" in why                                            # actionable, names the fix
    codes["tiingo"] = 200
    ok, why, prov = mk("t", "").equity_status()
    assert ok and prov.name == "tiingo"                                                  # Alpaca is not needed at all


def test_research_works_with_only_a_tiingo_key_and_reports_provider(tmp_path, monkeypatch):
    import research.providers as P
    from research.providers import TiingoProvider
    import pandas as pd
    import numpy as np
    from flask import Flask
    from research.api import init_research

    def fake_get(url, **kw):
        if "api.tiingo.com" in url:
            if url.endswith("/prices") and "startDate" in kw["params"] and "/SPY/" in url and kw["params"].get("format") == "json" and "endDate" not in kw["params"]:
                return _R(200)                                                           # key probe handled by requests.get below
            tk = url.split("/daily/")[1].split("/")[0]
            idx = pd.bdate_range(kw["params"]["startDate"], kw["params"]["endDate"])
            r = np.random.default_rng(sum(map(ord, tk))).standard_normal(len(idx))
            return _TR(200, [{"date": d.isoformat() + "Z", "adjClose": float(p)} for d, p in zip(idx, 100 * np.exp(np.cumsum(r) / 100))])
        raise AssertionError(url)
    monkeypatch.setattr(P, "_get", fake_get)
    monkeypatch.setattr(P.requests, "get", lambda url, **kw: _R(200))                     # key probe
    lit_ok(monkeypatch)
    fake = FakeGemini(generation=gen(hyp(predictor_variable="VIXCLS", predictor_transform="log_diff", outcome_variable="NVDA", direction="any",
                                         question="Does a VIX rise predict next-day NVIDIA returns?", control_variables=["SPY"])),
                      causal=CAUSAL, narrative=NARR)
    app = Flask(__name__)
    init_research(app, str(tmp_path / "t.db"), sync=True, reasoner=Reasoner(client=fake),
                  data=DataService(fred=StubFred(), tiingo=TiingoProvider(key="k"), alpaca=AlpacaProvider(key="", secret=""),
                                   cache_path=str(tmp_path / "t.db")))
    c = app.test_client()
    cat = c.get("/api/research/catalog").get_json()["providers"]
    assert cat["equity_data"]["usable"] and cat["equity_data"]["provider"] == "tiingo" and cat["tiingo"]["configured"]
    assert c.get("/api/research/health").get_json()["equity_data_usable"] is True
    rep = c.get(c.post("/api/research/runs", json={"mode": "autonomous", "max_hypotheses": 1}).get_json()["tracking_url"]).get_json()
    h = rep["hypotheses"][0]
    assert rep["status"] == "completed" and h["status"] != "rejected"
    provs = {v["series_id"]: v["provider"] for v in h["provenance"]["variables"]}
    assert provs["NVDA"] == "tiingo" and provs["VIXCLS"] == "stub"
