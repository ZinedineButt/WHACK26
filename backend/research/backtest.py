"""Pre-registered long/short backtest of the hypothesis as a trading rule.

Rule (fixed before any statistic is computed): hold  pos_t = s * sign(x_t)  from
the close of day t, earning the outcome return of day t+lag. s is the
pre-registered direction (+1 positive / -1 negative); for direction 'any' it is
the sign of the DEVELOPMENT-sample correlation. Costs are charged on turnover.
No parameter is fitted on the holdout. The holdout P&L is a descriptive
restatement of the same frozen rule and never feeds back into any tuning."""
import numpy as np

from . import criteria as C, registry as R, stats as S, validation as V

ANN = 252


def applicable(h):
    if h["outcome"]["transform"] != R.LOG_RETURN:
        return False, ("Backtest needs a tradable return outcome; this hypothesis predicts "
                       f"'{h['outcome']['transform']}', which is not a P&L series.")
    return True, None


def frozen_sign(h, dev_r):
    if h["direction"] == "positive":
        return 1
    if h["direction"] == "negative":
        return -1
    return 1 if dev_r >= 0 else -1


def _max_dd(cum_pct):
    eq = np.exp(np.asarray(cum_pct) / 100.0)
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0) * 100.0), (eq / peak - 1.0) * 100.0


def _metrics(net, gross, bh, pos, cost):
    n = len(net)
    if n < 2:
        return {"n": n}
    sd = float(np.std(net, ddof=1))
    traded = pos != 0
    hit = float(np.mean(gross[traded] > 0)) if traded.any() else None
    cum = np.cumsum(net)
    dd, _ = _max_dd(cum)
    bsd = float(np.std(bh, ddof=1))
    return {
        "n": int(n), "total_return_pct": float((np.exp(cum[-1] / 100) - 1) * 100),
        "ann_return_pct": float(np.mean(net) * ANN), "ann_vol_pct": float(sd * np.sqrt(ANN)),
        "sharpe": float(np.mean(net) / sd * np.sqrt(ANN)) if sd > 0 else None,
        "max_drawdown_pct": dd, "hit_rate": hit, "mean_daily_net_pct": float(np.mean(net)),
        "avg_daily_turnover": float(np.mean(np.abs(np.diff(pos, prepend=0)))), "cost_drag_ann_pct": float(np.sum(cost) / n * ANN),
        "buy_hold_total_return_pct": float((np.exp(np.cumsum(bh)[-1] / 100) - 1) * 100),
        "buy_hold_sharpe": float(np.mean(bh) / bsd * np.sqrt(ANN)) if bsd > 0 else None,
    }


def _boot_mean_ci(a, seed=C.SEED):
    rng = np.random.default_rng(seed)
    n, blk = len(a), C.BT_BOOT_BLOCK
    nb = int(np.ceil(n / blk))
    starts = rng.integers(0, n - blk + 1, (C.BT_BOOT_B, nb))
    idx = (starts[:, :, None] + np.arange(blk)).reshape(C.BT_BOOT_B, -1)[:, :n]
    m = a[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def simulate(x, y, lag, sign, cost_bps=C.COST_BPS):
    """Returns per-outcome-day arrays aligned to outcome rows lag..n-1."""
    pos_t = sign * np.sign(x[: len(x) - lag])
    yo = y[lag:]
    gross = pos_t * yo                                   # % log return
    turn = np.abs(np.diff(pos_t, prepend=0.0))
    cost = turn * (cost_bps / 100.0)                     # bps -> %
    return pos_t, yo, gross, gross - cost, cost


def run_backtest(df, h, dev_r=None, sign=None, cost_bps=C.COST_BPS):
    ok, why = applicable(h)
    if not ok:
        return {"applicable": False, "reason": why, "verdict": "not_run"}
    k = h["lag_days"]
    n = len(df)
    a, b = V.split_indices(n)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    if sign is None:
        sign = frozen_sign(h, dev_r if dev_r is not None else float(np.corrcoef(*S.lagged_pair(x[:b], y[:b], k))[0, 1]))
    pos, yo, gross, net, cost = simulate(x, y, k, sign, cost_bps)
    rows = np.arange(k, n)                                # outcome row indices
    dev_m, hold_m = rows < b, rows >= b
    seg = lambda m: _metrics(net[m], gross[m], yo[m], pos[m], cost[m])
    dev, hold = seg(dev_m), seg(hold_m)
    verdict, rule = "inconclusive", ""
    if hold.get("n", 0) < C.BT_MIN_N:
        rule = f"Holdout has {hold.get('n', 0)} days (< {C.BT_MIN_N}); too few for a backtest verdict."
    else:
        ci = _boot_mean_ci(net[hold_m])
        hold["mean_daily_net_ci_95"] = ci
        if hold["mean_daily_net_pct"] <= 0:
            verdict, rule = "fail", "Holdout net mean return is not positive after costs."
        elif ci[0] > 0 and (hold["hit_rate"] or 0) > 0.5:
            verdict, rule = "pass", "Holdout net return positive, block-bootstrap CI above 0, hit rate above 50%."
        else:
            rule = "Holdout net return is positive but the CI includes 0 or the hit rate is not above 50%."
    cum_net, cum_bh = np.cumsum(net), np.cumsum(yo)
    _, dd = _max_dd(cum_net)
    dates = [d.date().isoformat() for d in df.index[k:]]
    h0 = int(np.argmax(hold_m)) if hold_m.any() else None           # first holdout position
    base_n = cum_net[h0 - 1] if h0 else 0.0
    base_b = cum_bh[h0 - 1] if h0 else 0.0
    curve = []
    for i, (d, c, g, q, p) in enumerate(zip(dates, cum_net / 100, cum_bh / 100, dd, pos)):
        ho = h0 is not None and i >= h0
        curve.append({"date": d, "strategy": float((np.exp(c) - 1) * 100), "buy_hold": float((np.exp(g) - 1) * 100),
                      "strategy_holdout": float((np.exp((cum_net[i] - base_n) / 100) - 1) * 100) if ho else None,
                      "buy_hold_holdout": float((np.exp((cum_bh[i] - base_b) / 100) - 1) * 100) if ho else None,
                      "drawdown": float(q), "position": int(p), "sample": "holdout" if ho else "development"})
    return {
        "applicable": True, "rule": f"Hold {'+' if sign > 0 else '-'}sign(predictor) from close of day t; earn outcome return of day t+{k}; "
                                     f"{cost_bps:g} bps per unit of turnover.",
        "sign": int(sign), "sign_source": "pre-registered direction" if h["direction"] != "any" else "development-sample correlation",
        "lag": k, "cost_bps": cost_bps, "development": dev, "holdout": hold,
        "verdict": verdict, "verdict_rule": rule,
        "criteria": "holdout net mean>0 AND block-bootstrap 95% CI>0 AND hit rate>50%; fail if net mean<=0",
        "caveats": ["Daily close-to-close fills, no slippage beyond the stated cost, no borrow fees.",
                    "The holdout P&L restates the same frozen rule; nothing was tuned on it.",
                    "A positive backtest of a statistical association is not evidence of causation."],
        "equity_curve": curve,
    }


def peer_metrics(x, y, lag, sign, cost_bps=C.COST_BPS):
    """Full-sample frozen-rule stats for a peer asset (no tuning)."""
    pos, yo, gross, net, cost = simulate(x, y, lag, sign, cost_bps)
    m = _metrics(net, gross, yo, pos, cost)
    return {k: m.get(k) for k in ("n", "sharpe", "hit_rate", "total_return_pct", "max_drawdown_pct", "buy_hold_total_return_pct")}
