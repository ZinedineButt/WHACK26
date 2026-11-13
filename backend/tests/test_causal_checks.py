"""The four checks that used to be reported as not_run: forward-volatility timing + predictive test, negative control, event study."""
import numpy as np
import pandas as pd

from research import hstages as H


def _frame(n=700, seed=1, effect=0.6, forward=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.3 * y[t - 1] + effect * x[t - 1] + rng.normal() * 0.5
    df = pd.DataFrame({"x": x, "y": y}, index=idx)
    df["c_trailing_vol"] = pd.Series(y, index=idx).rolling(5).std().bfill()
    return df


def _h(lag=1, forward=True):
    return {"lag_days": lag, "direction": "any", "predictor": {"transform": "log_diff"},
            "outcome": {"transform": "fwd_logvol_5" if forward else "log_return"}}


def test_forward_granger_and_timing_run_instead_of_not_run():
    dev = _frame()
    g = H._forward_granger(dev, _h())
    t = H._forward_timing(dev, _h())
    assert g["id"] == "granger" and g["result"] in ("pass", "fail")
    assert t["id"] == "lead_lag" and t["result"] in ("pass", "fail", "inconclusive")
    assert "p_predictor" in g["metrics"] and {"r_future", "r_past"} <= set(t["metrics"])


def test_negative_control_quiet_for_unrelated_series_and_loud_for_related():
    dev = _frame()
    h = _h(forward=False)
    rng = np.random.default_rng(9)
    unrelated = pd.Series(rng.normal(size=len(dev)), index=dev.index)
    ok = H._negative_control(dev, h, {"neg_series": unrelated, "neg_label": "noise", "neg_id": "N"})
    assert ok["result"] == "pass"
    related = dev["y"].copy()
    bad = H._negative_control(dev, h, {"neg_series": related, "neg_label": "copy of outcome", "neg_id": "N"})
    assert bad["result"] == "fail"


def test_negative_control_not_run_without_series_and_never_fabricated():
    t = H._negative_control(_frame(), _h(), None)
    assert t["result"] == "not_run" and t["metrics"] == {}
    t2 = H._negative_control(_frame(), _h(), {"neg_series": None, "neg_note": "could not load"})
    assert t2["result"] == "not_run" and "could not load" in t2["explanation"]


def test_event_study_requires_live_and_real_event_days():
    dev = _frame()
    assert H._event_study(dev, _h(), None)["result"] == "not_run"
    assert H._event_study(dev, _h(), {"live": False})["result"] == "not_run"
    live = H._event_study(dev, _h(forward=False), {"live": True})        # 2022-2024 sample contains many FOMC days
    assert live["result"] in ("pass", "fail") and live["metrics"]["event_days_removed"] >= 5
    short = _frame(n=260).iloc[:10]
    assert H._event_study(short, _h(), {"live": True})["result"] == "not_run"


def test_stage2_marks_all_six_checks_with_extras():
    dev = _frame()
    h = {**_h(), "control_variables": [], "predictor": {"transform": "log_diff", "variable_id": "A", "label": "A"},
         "outcome": {"transform": "fwd_logvol_5", "variable_id": "B", "label": "B"}}
    prop = {"mechanisms": [], "alternative_explanations": [], "confounders": [], "assumptions": []}
    corr = {"association_verdict": "detected"}
    rng = np.random.default_rng(3)
    extras = {"live": True, "neg_series": pd.Series(rng.normal(size=len(dev)), index=dev.index), "neg_label": "noise", "neg_id": "N"}
    out = H.stage2(dev, h, prop, corr, extras=extras)
    res = {t["id"]: t["result"] for t in out["tests"]}
    assert set(res) >= {"lead_lag", "granger", "confounder_control", "placebo_shift", "negative_control", "event_study"}
    assert res["lead_lag"] != "not_run" and res["granger"] != "not_run"
    assert out["causal_verdict"] != "proven"
