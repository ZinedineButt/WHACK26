import numpy as np
import pandas as pd
import pytest

from research import pairs
from research.registry import Registry


class Svc:
    """Fake services: prices for four tickers. CA/CB are cointegrated (the gap pulls back); RA/RB are independent random walks."""
    def __init__(self, n=900, seed=5):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        common = np.cumsum(rng.normal(0, 1, n)) / 50
        gap = np.zeros(n)
        for t in range(1, n):
            gap[t] = 0.85 * gap[t - 1] + rng.normal(0, 0.02)      # strong mean reversion
        px = {"CA": np.exp(common + gap / 2), "CB": np.exp(common - gap / 2),
              "RA": np.exp(np.cumsum(rng.normal(0, 0.012, n))), "RB": np.exp(np.cumsum(rng.normal(0, 0.012, n)))}
        self.px = {k: pd.Series(100 * v, index=idx) for k, v in px.items()}
        self.registry = Registry(fred_available=True, equity_available=True)
        self.data = self


def levels(svc, v, years):
    return svc.px[v.id]


def test_cointegrated_pair_closes_and_is_flagged():
    r = pairs.run_pair(Svc(), "CA", "CB", years=3, entry=1.5, levels_fn=levels)
    assert r["events"]["total"] >= pairs.MIN_EVENTS
    assert r["horizons"][3]["all"]["mean"] > 0 and r["horizons"][3]["all"]["converged_share"] > 0.6   # 5-day horizon
    assert r["relationship"]["half_life_days"] is not None and r["relationship"]["half_life_days"] < 10
    assert r["relationship"]["adf_p"] < 0.05
    assert r["verdict"] in ("converges", "mixed")


def test_random_walks_show_no_reliable_closing():
    r = pairs.run_pair(Svc(), "RA", "RB", years=3, entry=1.5, levels_fn=levels)
    assert r["verdict"] != "converges"
    assert r["relationship"]["adf_p"] > 0.01


def test_no_lookahead_zscore_uses_only_past_window():
    s = Svc()
    full = pairs.run_pair(s, "CA", "CB", years=3, entry=2, levels_fn=levels)
    s.px["CA"].iloc[-1] *= 1.5                                    # change only the LAST day
    changed = pairs.run_pair(s, "CA", "CB", years=3, entry=2, levels_fn=levels)
    assert full["chart"]["z"][:-1] == changed["chart"]["z"][:-1]   # every earlier z is untouched


def test_input_validation():
    s = Svc()
    with pytest.raises(pairs.PairError):
        pairs.run_pair(s, "CA", "CA", levels_fn=levels)
    with pytest.raises(pairs.PairError):
        pairs.run_pair(s, "CA", "CB", entry=9, levels_fn=levels)
    with pytest.raises(pairs.PairError):
        pairs.run_pair(s, "CA", "not a ticker", levels_fn=levels)


# ---- second-level research, demo pairs, API and globe integration
from flask import Flask  # noqa: E402

from research import literature, pairs_deep  # noqa: E402
from research.api import init_research  # noqa: E402
from research.llm import Reasoner  # noqa: E402


def test_demo_pairs_show_both_outcomes_and_strength_scores():
    snap = pairs_deep.deepen(Svc(), pairs.run_pair(Svc(), "DEMO_SNAP_A", "DEMO_SNAP_B", 3, 2.0, levels), levels, [])
    rw = pairs_deep.deepen(Svc(), pairs.run_pair(Svc(), "DEMO_RW_A", "DEMO_RW_B", 3, 2.0, levels), levels, [])
    assert snap["synthetic"] and snap["verdict"] in ("converges", "mixed") and snap["strength"]["pct"] >= 60
    assert snap["placebo"]["p"] < 0.05 and snap["simulation"] is not None and snap["simulation"]["final_value"] > 0
    assert rw["verdict"] != "converges" and rw["strength"]["pct"] < snap["strength"]["pct"] and rw["simulation"] is None
    assert {c["id"] for c in snap["checks"]} == set(pairs_deep.WEIGHTS)


def test_demo_cannot_mix_with_real_assets():
    with pytest.raises(pairs.PairError):
        pairs.run_pair(Svc(), "DEMO_SNAP_A", "CA", levels_fn=levels)


@pytest.fixture
def client(tmp_path):
    app = Flask(__name__)
    init_research(app, str(tmp_path / "p.db"), sync=True, reasoner=Reasoner(api_key=""), data=Svc())
    return app.test_client()


def test_pairs_api_and_literature_with_papers_and_template_critique(client, monkeypatch):
    fake = [{"index": i, "title": f"Pairs trading paper {i}", "year": 2010 + i, "venue": "J", "url": f"u{i}", "cited_by": i, "abstract": "x",
             "doi": None, "source": "openalex"} for i in range(1, 4)]
    monkeypatch.setattr(literature, "search", lambda h, limit=5, query=None: {"status": "completed", "query": query, "provider": "openalex",
                                                                               "papers": fake, "note": ""})
    d = client.post("/api/research/pairs", json={"a": "DEMO_SNAP_A", "b": "DEMO_SNAP_B"}).get_json()
    assert d["synthetic"] and "checks" in d and "peer_pairs" in d
    lit = client.post("/api/research/pairs/literature", json={"a": "DEMO_SNAP_A", "b": "DEMO_SNAP_B"}).get_json()
    assert len(lit["literature"]["papers"]) == 3 and lit["critique"]["source"] == "template_demo" and lit["critique"]["critique"]
    assert client.post("/api/research/pairs", json={"a": "X", "b": "X"}).status_code == 400
    assert client.get("/api/research/pairs/suggest").get_json()["demo"]


def test_critique_drops_made_up_numbers():
    core = pairs_deep.deepen(Svc(), pairs.run_pair(Svc(), "DEMO_SNAP_A", "DEMO_SNAP_B", 3, 2.0, levels), levels, [])

    class Fake:
        configured = True

        def _call(self, prompt, schema):
            return {"confidence": "high", "comparison": "Gaps closed 12.34 percent of the time in the papers [1].",
                    "critique": [{"severity": "major", "point": "Costs may eat the edge."}], "paper_assessments": []}
    out = pairs_deep.critique_with_papers(Fake(), core, [{"index": 1, "title": "t", "year": 2010, "abstract": ""}])
    assert out["source"] == "template_fallback"                       # unsupported number 12.34 -> rejected, labelled fallback
