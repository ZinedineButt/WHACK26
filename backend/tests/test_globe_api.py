import math

import numpy as np
import pandas as pd
import pytest
from flask import Flask

from research import globe_api, literature
from research.api import init_research
from research.llm import Reasoner


class FakeData:
    """Synthetic equity prices where AAPL LEADS: MSFT follows AAPL's move the next day, TSLA does the opposite, NVDA is independent."""
    def __init__(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 400)
        prev = np.r_[0.0, a[:-1]]
        r = {"AAPL": a, "MSFT": 0.8 * prev + 0.6 * rng.normal(0, 1, 400), "TSLA": -0.6 * prev + 0.8 * rng.normal(0, 1, 400),
             "NVDA": rng.normal(0, 1, 400)}
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=400)
        self.px = {k: pd.Series(100 * np.exp(np.cumsum(v) / 100), index=idx) for k, v in r.items()}

    def equity_status(self):
        return True, "ok", None

    def get(self, variable, years):
        if variable.id not in self.px:
            from research.providers import ProviderError
            raise ProviderError("no bars")
        return self.px[variable.id], {}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_SYMBOLS", "AAPL,MSFT,TSLA,NVDA,AMD")
    globe_api._cache.clear()
    app = Flask(__name__)
    init_research(app, str(tmp_path / "t.db"), sync=True, reasoner=Reasoner(api_key=""), data=FakeData())
    return app.test_client()


def test_correlations_sign_size_and_order(client):
    d = client.get("/api/research/globe/correlations?symbol=AAPL&window=1y").get_json()
    assert d["status"] == "completed"
    by = {l["symbol"]: l for l in d["links"]}
    assert by["MSFT"]["r"] > 0.5 and by["TSLA"]["r"] < -0.3 and abs(by["NVDA"]["r"]) < 0.2
    assert "AMD" not in by and "AMD" in d["failed"]                    # unavailable symbol reported, not faked
    rs = [abs(l["r"]) for l in d["links"]]
    assert rs == sorted(rs, reverse=True)


def test_short_window_and_bad_input(client):
    assert client.get("/api/research/globe/correlations?symbol=AAPL&window=3m").get_json()["links"]
    assert client.get("/api/research/globe/correlations?symbol=not-a-ticker!").status_code == 400
    d = client.get("/api/research/globe/correlations?symbol=ZZZ").get_json()          # valid ticker, no data: honest "unavailable"
    assert d["status"] == "unavailable" and d["links"] == []
    assert client.get("/api/research/globe/correlations?symbol=AAPL&window=9y").status_code == 400


def test_papers_top5_trimmed(client, monkeypatch):
    fake = [{"index": i, "title": f"P{i}", "year": 2020, "venue": "J", "url": f"u{i}", "cited_by": i, "abstract": "x", "source": "openalex",
             "doi": None} for i in range(1, 6)]
    monkeypatch.setattr(literature, "search", lambda h, limit=5, query=None: {"status": "completed", "query": query, "provider": "openalex",
                                                                               "papers": fake, "note": ""})
    d = client.get("/api/research/globe/papers?symbol=AAPL").get_json()
    assert len(d["papers"]) == 5 and "abstract" not in d["papers"][0] and "Apple" in d["query"]


def test_network_all_pairs_sorted(client):
    d = client.get("/api/research/globe/network?window=1y").get_json()
    assert d["status"] == "completed"
    pairs = {frozenset((l["a"], l["b"])): l["r"] for l in d["links"]}
    assert len(pairs) == 6                                            # 4 available symbols -> 6 pairs, none duplicated
    assert pairs[frozenset(("AAPL", "MSFT"))] > 0.5 and pairs[frozenset(("AAPL", "TSLA"))] < -0.3
    rs = [abs(l["r"]) for l in d["links"]]
    assert rs == sorted(rs, reverse=True)
    assert client.get("/api/research/globe/network?window=9y").status_code == 400


def test_any_map_symbol_works_via_symbols_param(client):
    q = "symbols=AAPL,MSFT,TSLA,PLTR,WMT,SHOP,KO,bad!sym"
    d = client.get(f"/api/research/globe/correlations?symbol=MSFT&window=1y&{q}").get_json()
    assert d["status"] == "completed" and {l["symbol"] for l in d["links"]} == {"AAPL", "TSLA"}
    assert set(d["failed"]) == {"PLTR", "WMT", "SHOP", "KO"}                          # no data -> reported, never faked
    n = client.get(f"/api/research/globe/network?window=1y&{q}").get_json()
    assert n["symbols_used"] == ["AAPL", "MSFT", "TSLA"] and len(n["links"]) == 3
    # a selected symbol that is not in the list is still analysed against the list
    d2 = client.get("/api/research/globe/correlations?symbol=NVDA&symbols=AAPL,MSFT").get_json()
    assert d2["status"] == "completed" and len(d2["links"]) == 2


def test_papers_uses_company_name(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(literature, "search", lambda h, limit=5, query=None: seen.setdefault("q", query) and {
        "status": "no_results", "query": query, "provider": None, "papers": [], "note": "x"})
    client.get("/api/research/globe/papers?symbol=PLTR&name=Palantir Technologies")
    assert "Palantir" in seen["q"]


COMPANIES = [{"symbol": "AAPL", "company": "Apple"}, {"symbol": "MSFT", "company": "Microsoft"},
             {"symbol": "TSLA", "company": "Tesla"}, {"symbol": "NVDA", "company": "NVIDIA"}]


def test_ask_keyword_fallback_uses_real_numbers(client):
    d = client.post("/api/research/globe/ask", json={"question": "does Apple move with Tesla?", "companies": COMPANIES}).get_json()
    assert d["source"] == "keyword_match" and d["status"] == "completed"
    p = d["pairs"][0]
    assert {p["a"], p["b"]} == {"AAPL", "TSLA"} and p["r"] < -0.3 and p["direction"] == "negative" and p["matches_expectation"] is None


def test_ask_gemini_pairs_validated_and_scored(tmp_path, monkeypatch):
    import json as _json

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                class R:
                    text = _json.dumps({"kind": "lead_lag", "research_question": "Does Apple lead Microsoft?", "pairs": [
                        {"a": "AAPL", "b": "MSFT", "expected": "positive", "reason": "Both are large platform companies."},
                        {"a": "AAPL", "b": "ZZZZ", "expected": "negative", "reason": "off the map"},
                        {"a": "AAPL", "b": "TSLA", "expected": "positive", "reason": "wrong guess"}]})
                return R

    monkeypatch.setenv("ALPACA_SYMBOLS", "AAPL,MSFT,TSLA,NVDA")
    globe_api._cache.clear()
    app = Flask(__name__)
    init_research(app, str(tmp_path / "t.db"), sync=True, reasoner=Reasoner(api_key="k", client=FakeClient), data=FakeData())
    d = app.test_client().post("/api/research/globe/ask", json={"question": "which tech stocks move together", "companies": COMPANIES}).get_json()
    assert d["source"] == "gemini" and len(d["pairs"]) == 2                       # ZZZZ dropped
    by = {p["b"]: p for p in d["pairs"]}
    assert by["MSFT"]["matches_expectation"] is True and by["TSLA"]["matches_expectation"] is False
    assert d["research_question"].startswith("Does Apple")


def test_ask_validation(client):
    assert client.post("/api/research/globe/ask", json={"question": "hi", "companies": COMPANIES}).status_code == 400
    assert client.post("/api/research/globe/ask", json={"question": "does this work", "companies": []}).status_code == 400


def test_replay_real_moves_and_peers(client):
    d = client.get("/api/research/globe/replay?days=10").get_json()
    assert d["status"] == "completed" and len(d["days"]) == 11
    assert set(d["moves"]) == {"AAPL", "MSFT", "TSLA", "NVDA"} and all(len(v) == 11 for v in d["moves"].values())
    assert d["peers"]["AAPL"][0]["symbol"] == "MSFT" and d["peers"]["AAPL"][0]["r"] > 0.5
    assert all(abs(p - (math.exp(m / 100) - 1) * 100) < 0.01 for p, m in zip(d["pct"]["AAPL"], d["moves"]["AAPL"]))   # simple % change
    assert len(d["peers"]["AAPL"]) == 3 and client.get("/api/research/globe/replay?days=x").status_code == 400
    assert set(d["close"]) == set(d["moves"]) and all(len(v) == 11 for v in d["close"].values())                 # real closes for zoom-in view
    assert all(c is None or c > 0 for v in d["close"].values() for c in v)