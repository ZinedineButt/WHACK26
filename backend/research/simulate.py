"""Paper-trade simulation of a completed hypothesis's frozen strategy against a
benchmark (holding the S&P 500). It replays stored research data only: no
parameter is re-fitted, and no orders are ever placed anywhere."""
import numpy as np
import pandas as pd

from . import backtest as BT, criteria as C

ANN = 252
PERIODS = ("holdout", "development", "full")
MIN_CAPITAL, MAX_CAPITAL = 100.0, 10_000_000.0


class SimulationError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def benchmark_variable_id(mode, alpaca_usable):
    if mode == "demo_autonomous":
        return "SYN_MKT"
    return "SPY" if alpaca_usable else "SP500"


def _curve_metrics(net_log_pct, capital, n):
    tot = float(np.sum(net_log_pct))
    final = capital * float(np.exp(tot / 100))
    sd = float(np.std(net_log_pct, ddof=1)) if n > 1 else 0.0
    cum = np.cumsum(net_log_pct)
    eq = np.exp(cum / 100)
    dd = float(np.min(eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1) * 100) if n else 0.0
    return {"final_value": final, "total_return_pct": float((np.exp(tot / 100) - 1) * 100),
            "cagr_pct": float((np.exp(tot / 100 * (ANN / n)) - 1) * 100) if n >= 20 else None,
            "ann_vol_pct": float(sd * np.sqrt(ANN)), "sharpe": float(np.mean(net_log_pct) / sd * np.sqrt(ANN)) if sd > 0 else None,
            "max_drawdown_pct": dd, "best_day_pct": float(np.max(net_log_pct)), "worst_day_pct": float(np.min(net_log_pct))}


def run_simulation(rep, hid, benchmark_levels, benchmark_label, capital=10_000.0, period="holdout", cost_bps=C.COST_BPS):
    h = next((x for x in rep.get("hypotheses", []) if x["hypothesis_id"] == hid), None)
    if h is None:
        raise SimulationError(f"unknown hypothesis '{hid}'", 404)
    bt = h.get("backtest")
    if h["status"] != "completed" or not bt:
        raise SimulationError("Only hypotheses that completed the full pipeline have a strategy to simulate "
                              f"(this one is '{h['status']}').", 409)
    if not bt.get("applicable"):
        raise SimulationError("No simulation: " + bt.get("reason", "the outcome is not a tradable return."), 400)
    ts = (rep.get("visualization_data", {}).get(hid) or {}).get("timeseries")
    if not ts:
        raise SimulationError("Stored research data for this hypothesis is missing.", 409)
    k, sign = int(bt["lag"]), int(bt["sign"])
    dates = pd.to_datetime([r["date"] for r in ts])
    x = np.array([r["predictor"] for r in ts], float)
    y = np.array([r["outcome"] for r in ts], float)
    sample = np.array([r["sample"] for r in ts])

    pos, yo, gross, net, cost = BT.simulate(x, y, k, sign, cost_bps)
    odates, osample = dates[k:], sample[k:]
    mask = np.ones(len(odates), bool) if period == "full" else (osample == period)
    if mask.sum() < 20:
        raise SimulationError(f"Only {int(mask.sum())} days in the '{period}' period; need at least 20.", 400)

    lv = benchmark_levels.reindex(dates, method="ffill")
    warnings = []
    if lv.isna().any():
        lv = lv.bfill()
        warnings.append("Benchmark data did not cover the full period; early gaps were back-filled.")
    bench_all = np.r_[np.nan, 100 * np.diff(np.log(lv.to_numpy(float)))]
    bench = bench_all[k:]
    nan_n = int(np.isnan(bench[mask]).sum())
    if nan_n:
        bench = np.nan_to_num(bench, nan=0.0)
        warnings.append(f"{nan_n} benchmark day(s) had no return and were treated as 0%.")

    nm, ym, bm, pm = net[mask], yo[mask], bench[mask], pos[mask]
    dm = odates[mask]
    n = int(mask.sum())
    cs, ca, cb = np.cumsum(nm), np.cumsum(ym), np.cumsum(bm)
    curve = [{"date": dm[0].date().isoformat(), "strategy": capital, "asset": capital, "benchmark": capital, "position": int(pm[0])}]
    curve += [{"date": d.date().isoformat(), "strategy": float(capital * np.exp(a / 100)), "asset": float(capital * np.exp(b / 100)),
               "benchmark": float(capital * np.exp(c / 100)), "position": int(p)} for d, a, b, c, p in zip(dm, cs, ca, cb, pm)]

    ms, ma, mb = (_curve_metrics(v, capital, n) for v in (nm, ym, bm))
    active = nm - bm
    ci = BT._boot_mean_ci(active) if n >= C.BT_MIN_N else None
    beta = float(np.cov(nm, bm)[0, 1] / np.var(bm, ddof=1)) if np.var(bm) > 0 else None
    excess = ms["total_return_pct"] - mb["total_return_pct"]
    if ms["final_value"] > mb["final_value"] and ci and ci[0] > 0:
        verdict, why = "outperformed", "The strategy ended ahead and the bootstrap CI of the mean daily active return is above 0."
    elif ms["final_value"] <= mb["final_value"]:
        verdict, why = "underperformed", "The strategy ended behind the benchmark."
    else:
        verdict, why = "inconclusive", ("The strategy ended ahead, but the bootstrap CI of the mean daily active return includes 0 "
                                        "(or the period is too short), so the lead may be luck.")
    last_signal = int(sign * np.sign(x[-1]))
    return {
        "hypothesis_id": hid, "question": h["normalized"]["question"], "period": period, "capital": capital, "cost_bps": cost_bps,
        "start": dm[0].date().isoformat(), "end": dm[-1].date().isoformat(), "n_days": n,
        "rule": bt["rule"], "benchmark": {"variable_id": benchmark_label["variable_id"], "label": benchmark_label["label"],
                                          "note": benchmark_label["note"]},
        "strategy": ms, "asset_buy_hold": {**ma, "label": h["normalized"]["outcome"]["label"]}, "benchmark_buy_hold": mb,
        "vs_benchmark": {"excess_return_pct": float(excess), "excess_final_value": float(ms["final_value"] - mb["final_value"]),
                         "mean_daily_active_return_pct": float(np.mean(active)), "active_return_ci_95": ci,
                         "beta": beta, "correlation": float(np.corrcoef(nm, bm)[0, 1]) if np.std(nm) > 0 and np.std(bm) > 0 else None,
                         "verdict": verdict, "verdict_rule": why},
        "current_signal": {"as_of": dates[-1].date().isoformat(), "position": last_signal,
                           "meaning": (("Long" if last_signal > 0 else "Short" if last_signal < 0 else "Flat") +
                                       f" {h['normalized']['outcome']['label']} for day t+{k} under the frozen rule"),
                           "note": "What the rule would hold after the latest observation. Not investment advice; no order is placed."},
        "period_note": {"holdout": "Out-of-sample: the sealed final 25% that the rule was never tuned on.",
                        "development": "In-sample: the period the hypothesis was discovered on, so flattering by construction.",
                        "full": "Mixes in-sample and out-of-sample days; treat as optimistic."}[period],
        "caveats": ["Simulation only: daily close-to-close fills, the stated cost per unit of turnover, no slippage, borrow fees or taxes.",
                    "Past performance of a statistical rule does not predict future returns; a rule can work once and fail afterwards.",
                    "The benchmark is a price index/ETF held without rebalancing.", *warnings],
        "curve": curve,
    }
