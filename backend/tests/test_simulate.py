"""Strategy simulation against a benchmark."""
import json

import numpy as np
import pandas as pd
import pytest

import research.engine as E
from research.llm import Reasoner
from research.providers import ProviderError
from research.simulate import SimulationError, benchmark_variable_id, run_simulation
from tests.test_autonomous import CAUSAL, NARR, FakeGemini, gen, hyp, make

LABEL = {"variable_id": "SPY", "label": "S&P 500 (buy and hold)", "note": "test"}


def craft_report(n=60, lag=1, sign=1):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    dates = pd.bdate_range("2024-01-02", periods=n)
    samples = ["development"] * 40 + ["holdout"] * (n - 40)
    ts = [{"date": d.date().isoformat(), "predictor": float(a), "outcome": float(b), "sample": s}
          for d, a, b, s in zip(dates, x, y, samples)]
    rep = {"hypotheses": [{"hypothesis_id": "H1", "status": "completed", "normalized": {"question": "q", "outcome": {"label": "Asset"}},
                           "backtest": {"applicable": True, "lag": lag, "sign": sign, "rule": "rule"}}],
           "visualization_data": {"H1": {"timeseries": ts}}}
    return rep, x, y, dates


def levels_for(dates, rets_pct):
    return pd.Series(100 * np.exp(np.cumsum(rets_pct) / 100), index=dates)


def test_simulation_math_matches_hand_calculation():
    rep, x, y, dates = craft_report()
    out = run_simulation(rep, "H1", levels_for(dates, np.full(len(dates), 0.1)), LABEL, capital=10_000, period="holdout", cost_bps=10)
    pos = np.sign(x[:-1])
    net = pos * y[1:] - np.abs(np.diff(pos, prepend=0.0)) * 0.1
    odates = dates[1:]
    m = np.array([d >= dates[40] for d in odates])
    assert out["n_days"] == int(m.sum())
    assert out["strategy"]["final_value"] == pytest.approx(10_000 * np.exp(net[m].sum() / 100))
    assert out["asset_buy_hold"]["final_value"] == pytest.approx(10_000 * np.exp(y[1:][m].sum() / 100))
    assert out["benchmark_buy_hold"]["final_value"] == pytest.approx(10_000 * np.exp(0.1 * m.sum() / 100))
    assert out["curve"][0]["strategy"] == 10_000 and out["curve"][-1]["strategy"] == pytest.approx(out["strategy"]["final_value"])
    assert out["vs_benchmark"]["excess_final_value"] == pytest.approx(out["strategy"]["final_value"] - out["benchmark_buy_hold"]["final_value"])
    assert out["start"] == odates[m][0].date().isoformat() and out["end"] == odates[-1].date().isoformat()


def test_periods_select_the_right_days_and_sign_flips_the_trade():
    rep, _, _, dates = craft_report()
    b = levels_for(dates, np.zeros(len(dates)))
    full = run_simulation(rep, "H1", b, LABEL, period="full")
    hold = run_simulation(rep, "H1", b, LABEL, period="holdout")
    dev = run_simulation(rep, "H1", b, LABEL, period="development")
    assert full["n_days"] == hold["n_days"] + dev["n_days"] == len(dates) - 1
    assert "never tuned" in hold["period_note"] and "flattering" in dev["period_note"]
    neg_rep, *_ = craft_report(sign=-1)
    neg = run_simulation(neg_rep, "H1", b, LABEL, period="full", cost_bps=0)
    pos = run_simulation(rep, "H1", b, LABEL, period="full", cost_bps=0)
    assert np.log1p(neg["strategy"]["total_return_pct"] / 100) == pytest.approx(-np.log1p(pos["strategy"]["total_return_pct"] / 100), abs=1e-9)


def test_costs_lower_the_result_and_flat_benchmark_has_zero_return():
    rep, _, _, dates = craft_report()
    b = levels_for(dates, np.zeros(len(dates)))
    free = run_simulation(rep, "H1", b, LABEL, period="full", cost_bps=0)
    pricey = run_simulation(rep, "H1", b, LABEL, period="full", cost_bps=50)
    assert pricey["strategy"]["final_value"] < free["strategy"]["final_value"]
    assert free["benchmark_buy_hold"]["total_return_pct"] == pytest.approx(0, abs=1e-9)


def test_verdict_needs_ci_above_zero_not_just_a_higher_final_value():
    rep, _, _, dates = craft_report(n=70)
    out = run_simulation(rep, "H1", levels_for(dates, np.full(len(dates), -0.5)), LABEL, period="full", cost_bps=0)
    ahead = out["strategy"]["final_value"] > out["benchmark_buy_hold"]["final_value"]
    ci = out["vs_benchmark"]["active_return_ci_95"]
    if ahead and ci[0] <= 0:
        assert out["vs_benchmark"]["verdict"] == "inconclusive"
    if not ahead:
        assert out["vs_benchmark"]["verdict"] == "underperformed"


def test_current_signal_uses_latest_observation_only():
    rep, x, _, dates = craft_report()
    out = run_simulation(rep, "H1", levels_for(dates, np.zeros(len(dates))), LABEL, period="full")
    assert out["current_signal"]["meaning"].startswith("Long" if x[-1] > 0 else "Short")
    assert out["current_signal"]["as_of"] == dates[-1].date().isoformat() and "no order is placed" in out["current_signal"]["note"]


@pytest.mark.parametrize("mut,status", [(lambda r: r["hypotheses"][0].update(status="rejected"), 409),
                                         (lambda r: r["hypotheses"][0]["backtest"].update(applicable=False, reason="vol"), 400),
                                         (lambda r: r["visualization_data"].clear(), 409)])
def test_not_simulatable_cases(mut, status):
    rep, _, _, dates = craft_report()
    mut(rep)
    with pytest.raises(SimulationError) as e:
        run_simulation(rep, "H1", levels_for(dates, np.zeros(len(dates))), LABEL)
    assert e.value.status == status


def test_too_short_period_rejected():
    rep, _, _, dates = craft_report(n=60)
    for r in rep["visualization_data"]["H1"]["timeseries"][50:]:
        r["sample"] = "development"
    with pytest.raises(SimulationError, match="at least 20"):
        run_simulation(rep, "H1", levels_for(dates, np.zeros(len(dates))), LABEL, period="holdout")


def test_benchmark_choice():
    assert benchmark_variable_id("autonomous", True) == "SPY" and benchmark_variable_id("autonomous", False) == "SP500"
    assert benchmark_variable_id("demo_autonomous", True) == "SYN_MKT"


# ------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def demo_client(tmp_path_factory):
    c = make(tmp_path_factory.mktemp("sim"))
    rid = c.post("/api/research/runs", json={"mode": "demo_autonomous"}).get_json()["run_id"]
    return c, rid


def test_api_matches_backtest_and_uses_synthetic_benchmark(demo_client):
    c, rid = demo_client
    rep = c.get(f"/api/research/runs/{rid}").get_json()
    r = c.post(f"/api/research/runs/{rid}/simulate", json={"hypothesis_id": "H1", "capital": 25000})
    assert r.status_code == 200
    d = r.get_json()
    assert d["benchmark"]["variable_id"] == "SYN_MKT" and "SYNTHETIC" in d["benchmark"]["note"]
    assert d["strategy"]["total_return_pct"] == pytest.approx(rep["hypotheses"][0]["backtest"]["holdout"]["total_return_pct"], rel=1e-6)
    assert d["capital"] == 25000 and d["curve"][0]["strategy"] == 25000
    json.dumps(d, allow_nan=False)


@pytest.mark.parametrize("body,status,frag", [
    ({"hypothesis_id": "H2"}, 409, "completed the full pipeline"),
    ({"hypothesis_id": "H1", "capital": 5}, 400, "capital"),
    ({"hypothesis_id": "H1", "capital": True}, 400, "capital"),
    ({"hypothesis_id": "H1", "period": "future"}, 400, "period"),
    ({"hypothesis_id": "H1", "cost_bps": 500}, 400, "cost_bps"),
    ({"hypothesis_id": "NOPE"}, 404, "unknown hypothesis"),
    ({}, 400, "hypothesis_id"),
    ({"hypothesis_id": "H1", "orders": 1}, 400, "body must be"),
])
def test_api_validation(demo_client, body, status, frag):
    c, rid = demo_client
    r = c.post(f"/api/research/runs/{rid}/simulate", json=body)
    assert r.status_code == status and frag in r.get_json()["error"], r.get_json()


def test_api_unknown_legacy_and_unfinished_runs(tmp_path):
    c = make(tmp_path)
    assert c.post("/api/research/runs/nope/simulate", json={"hypothesis_id": "H1"}).status_code == 404
    legacy = c.post("/api/research/runs", json={"preset_id": "demo_effect"}).get_json()["run_id"]
    assert c.post(f"/api/research/runs/{legacy}/simulate", json={"hypothesis_id": "H1"}).status_code == 400
    from research.engine import new_report
    c.application.extensions["research"]["store"].create("r1", "autonomous", new_report("r1", "autonomous", {}))
    assert c.post("/api/research/runs/r1/simulate", json={"hypothesis_id": "H1"}).status_code == 409


def test_live_run_uses_fred_sp500_benchmark_and_reports_provider_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(E.LIT, "search", lambda h, **kw: {"status": "no_results", "query": "q", "provider": "openalex", "note": "", "papers": []})
    c = make(tmp_path, Reasoner(client=FakeGemini(generation=gen(hyp()), causal=CAUSAL, narrative=NARR)))
    rid = c.post("/api/research/runs", json={"mode": "autonomous", "max_hypotheses": 1}).get_json()["run_id"]
    assert c.get(f"/api/research/runs/{rid}").get_json()["hypotheses"][0]["status"] == "completed"
    d = c.post(f"/api/research/runs/{rid}/simulate", json={"hypothesis_id": "H1", "period": "full"}).get_json()
    assert d["benchmark"]["variable_id"] == "SP500" and "price return only" in d["benchmark"]["note"]
    svc = c.application.extensions["research"]["services"]

    def boom(*a, **k):
        raise ProviderError("provider down")
    monkeypatch.setattr(svc.data, "get", boom)
    r = c.post(f"/api/research/runs/{rid}/simulate", json={"hypothesis_id": "H1"})
    assert r.status_code == 502 and "Benchmark data unavailable" in r.get_json()["error"]
