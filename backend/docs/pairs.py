"""Pairs divergence study: when asset X and asset Y drift apart, do they come back together (statistical arbitrage)?

Everything is computed from real daily prices; the AI only suggests which pairs to look at and never produces a number.

Method (no look-ahead anywhere):
  spread_t   = 100 * (ln X_t - ln Y_t)                  relative performance of X vs Y
  gap z_t    = (spread_t - mean of the last 60 days) / std of the last 60 days    (both known at the close of day t)
  event      = first day |z| crosses the entry threshold (default 2)
  reversion  = -sign(z_t) * (spread_{t+h} - spread_t)    > 0 means the gap closed; measured at h = 1, 2, 3, 5, 10, 20 days
So "nothing happens the next day" and "closes later" are both visible in the horizon profile.
The best horizon is chosen on the first 70% of the history and judged ONCE on the last 30%.
A simple trading rule (enter at |z| >= entry, exit when |z| < 0.5 or after 20 days, 5 bp cost per leg per side) is replayed
on the out-of-sample part. Cointegration (Engle-Granger) and the mean-reversion half-life are full-sample descriptions only."""
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import registry as R
from . import transforms as T
from .providers import ProviderError

WINDOW = 60
HORIZONS = (1, 2, 3, 5, 10, 20)
EXIT_Z = 0.5
MAX_HOLD = 20
COST_BPS_PER_LEG = 5.0
TRAIN_FRAC = 0.7
MIN_DAYS = 250
MIN_EVENTS = 8


NAMES = {"KO": "Coca-Cola", "PEP": "PepsiCo", "V": "Visa", "MA": "Mastercard", "XOM": "Exxon Mobil", "CVX": "Chevron",
         "GLD": "Gold (SPDR GLD)", "SLV": "Silver (iShares SLV)", "HD": "Home Depot", "LOW": "Lowe's", "JPM": "JPMorgan Chase",
         "BAC": "Bank of America", "WMT": "Walmart", "TGT": "Target", "DAL": "Delta Air Lines", "UAL": "United Airlines",
         "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF", "NVDA": "NVIDIA", "AMD": "AMD", "AAPL": "Apple", "MSFT": "Microsoft",
         "GOOGL": "Alphabet", "META": "Meta", "UPS": "UPS", "FDX": "FedEx", "MCD": "McDonald's", "SBUX": "Starbucks",
         "DCOILWTICO": "WTI crude oil", "DCOILBRENTEU": "Brent crude oil", "SP500": "S&P 500 index", "NASDAQ100": "Nasdaq-100 index"}


NAMES.update({"DEMO_SNAP_A": "Demo asset A (synthetic)", "DEMO_SNAP_B": "Demo asset B (synthetic)",
              "DEMO_RW_A": "Demo asset C (synthetic)", "DEMO_RW_B": "Demo asset D (synthetic)",
              "DEMO_SNAP2_A": "Demo asset E (synthetic)", "DEMO_SNAP2_B": "Demo asset F (synthetic)"})
# Synthetic demo pairs: two whose gap snaps back, one pair of unrelated random walks (so the demo shows both outcomes).
DEMO_PAIRS = [("DEMO_SNAP_A", "DEMO_SNAP_B", "Synthetic pair whose gap is built to pull back."),
              ("DEMO_RW_A", "DEMO_RW_B", "Synthetic pair of unrelated random walks: no pull-back exists."),
              ("DEMO_SNAP2_A", "DEMO_SNAP2_B", "A second synthetic pair with a slower pull-back.")]
_DEMO_PARAMS = {"DEMO_SNAP": (0.88, 11), "DEMO_SNAP2": (0.96, 23), "DEMO_RW": (None, 37)}


def _demo_prices(vid):
    """Deterministic SYNTHETIC prices (1500 business days). SNAP: shared trend + a mean-reverting gap. RW: independent walks."""
    stem, leg = vid.rsplit("_", 1)
    phi, seed = _DEMO_PARAMS[stem]
    rng = np.random.default_rng(seed)
    n = 1500
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    if phi is None:
        lvl = np.cumsum(rng.normal(0, 0.012, n)) if leg == "A" else np.cumsum(np.random.default_rng(seed + 1).normal(0, 0.012, n))
        return pd.Series(100 * np.exp(lvl), index=idx)
    common = np.cumsum(rng.normal(0, 0.01, n))
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = phi * gap[t - 1] + rng.normal(0, 0.012)
    return pd.Series(100 * np.exp(common + (gap / 2 if leg == "A" else -gap / 2)), index=idx)


class PairError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


class _DemoVar:
    kind, source = "price", "demo"

    def __init__(self, vid):
        self.id, self.label = vid, NAMES.get(vid, vid)


def _prices(svc, vid, years, levels_fn):
    if vid.startswith("DEMO_"):
        stem = vid.rsplit("_", 1)[0]
        if stem not in _DEMO_PARAMS or vid[-1] not in "AB":
            raise PairError(f"{vid}: unknown demo asset")
        return _DemoVar(vid), _demo_prices(vid)
    v, why = svc.registry.resolve(vid)
    if v is None:
        raise PairError(f"{vid}: {why}")
    if v.kind != "price":
        raise PairError(f"{vid} is not a price series (use a stock, ETF, index, commodity or FX price)")
    try:
        s = levels_fn(svc, v, years) if v.source == "equity" else svc.data.get(v, years)[0]
    except ProviderError as e:
        raise PairError(f"{vid}: {e}", 502)
    s = s[s > 0].dropna()
    return v, s


def _event_days(z, entry):
    """First day of each excursion beyond +-entry (consecutive days beyond the threshold count once)."""
    beyond = z.abs() >= entry
    return list(z.index[beyond & ~beyond.shift(1, fill_value=False)])


def _reversion(spread, z, idx, h):
    """-sign(z)*(spread_{t+h}-spread_t) at integer positions idx (needs t+h inside the sample)."""
    pos = [i for i in idx if i + h < len(spread)]
    if not pos:
        return np.array([])
    s, zz = spread.to_numpy(), z.to_numpy()
    return np.array([-np.sign(zz[i]) * (s[i + h] - s[i]) for i in pos])


def _summ(x):
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "converged_share": None, "t": None, "p": None}
    mean = float(np.mean(x))
    if n > 2 and np.std(x, ddof=1) > 0:
        t = mean / (np.std(x, ddof=1) / math.sqrt(n))
        p = float(2 * (1 - sps.t.cdf(abs(t), n - 1)))
    else:
        t, p = None, None
    return {"n": n, "mean": round(mean, 4), "median": round(float(np.median(x)), 4),
            "converged_share": round(float(np.mean(x > 0)), 4), "t": None if t is None else round(float(t), 2),
            "p": None if p is None else round(p, 4)}


def _trades(spread, z, entry, start, end, cost_bps=COST_BPS_PER_LEG):
    """Replay the rule between integer positions [start, end). Returns net P&L per trade in spread points (% of one leg)."""
    s, zz = spread.to_numpy(), z.to_numpy()
    out, i = [], max(start, 1)
    cost = 4 * cost_bps / 100.0               # 2 legs x (enter + exit), in percentage points
    while i < end - 1:
        if abs(zz[i]) >= entry and abs(zz[i - 1]) < entry:
            side, t0 = -np.sign(zz[i]), i
            j = i + 1
            while j < end - 1 and abs(zz[j]) >= EXIT_Z and j - t0 < MAX_HOLD:
                j += 1
            out.append({"entry": str(spread.index[t0].date()), "days": int(j - t0),
                        "net": float(side * (s[j] - s[t0]) - cost)})
            i = j + 1
        else:
            i += 1
    return out


def _coint_halflife(spread):
    from statsmodels.tsa.stattools import adfuller
    res = {"adf_p": None, "half_life_days": None, "phi": None}
    s = spread.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res["adf_p"] = round(float(adfuller(s, autolag="AIC")[1]), 4)
        except Exception:
            pass
    x, y = s[:-1], s[1:]
    if np.std(x) > 0:
        phi = float(np.polyfit(x - x.mean(), y - y.mean(), 1)[0])
        res["phi"] = round(phi, 4)
        if 0 < phi < 1:
            res["half_life_days"] = round(-math.log(2) / math.log(phi), 1)
    return res


def run_pair(svc, a, b, years=3, entry=2.0, levels_fn=None):
    a, b = a.strip().upper(), b.strip().upper()
    if a == b:
        raise PairError("choose two different assets")
    if not (1 <= entry <= 3.5):
        raise PairError("entry threshold must be between 1 and 3.5 standard deviations")
    years = max(1, min(10, float(years)))
    if a.startswith("DEMO_") != b.startswith("DEMO_"):
        raise PairError("demo assets can only be paired with other demo assets")
    va, pa = _prices(svc, a, years, levels_fn)
    vb, pb = _prices(svc, b, years, levels_fn)
    df = pd.concat([pa.rename("a"), pb.rename("b")], axis=1, join="inner").dropna()
    if len(df) < MIN_DAYS:
        raise PairError(f"only {len(df)} overlapping trading days; need at least {MIN_DAYS}")
    spread = 100 * (np.log(df["a"]) - np.log(df["b"]))
    mu, sd = spread.rolling(WINDOW).mean(), spread.rolling(WINDOW).std(ddof=1)
    z = ((spread - mu) / sd.where(sd > 0)).dropna()
    spread = spread.loc[z.index]
    n = len(z)
    split = int(n * TRAIN_FRAC)
    events = _event_days(z, entry)
    pos_all = [z.index.get_loc(d) for d in events]
    train_pos = [p for p in pos_all if p < split]
    test_pos = [p for p in pos_all if p >= split]

    rows = []
    for h in HORIZONS:
        full = _summ(_reversion(spread, z, pos_all, h))
        train = _summ(_reversion(spread, z, train_pos, h))
        test = _summ(_reversion(spread, z, test_pos, h))
        rows.append({"horizon_days": h, "all": full, "train": train, "test": test})
    # control: ordinary days with a tiny gap should show no reversion
    calm = [i for i in range(n) if abs(z.iloc[i]) < 0.5 and abs(z.iloc[i]) > 0.0]
    control = {h: _summ(_reversion(spread, z, calm, h))["mean"] for h in (1, 5, 10)}

    ok = [r for r in rows if r["train"]["n"] >= 5 and r["train"]["mean"] is not None]
    best = max(ok, key=lambda r: r["train"]["mean"]) if ok else None
    best_h = best["horizon_days"] if best else None
    oos = next((r for r in rows if r["horizon_days"] == best_h), None)["test"] if best else None

    trades_all = _trades(spread, z, entry, 0, n)
    trades_test = _trades(spread, z, entry, split, n)
    ts = lambda tr: {"n": len(tr), "avg_net": round(float(np.mean([t["net"] for t in tr])), 4) if tr else None,
                     "win_rate": round(float(np.mean([t["net"] > 0 for t in tr])), 4) if tr else None,
                     "avg_days": round(float(np.mean([t["days"] for t in tr])), 1) if tr else None,
                     "total_net": round(float(np.sum([t["net"] for t in tr])), 3) if tr else None}
    stat = _coint_halflife(spread)

    verdict, why = "no_evidence", "Gaps did not reliably close, or there were too few divergences to tell."
    if len(events) < MIN_EVENTS:
        verdict, why = "inconclusive", f"Only {len(events)} divergences beyond {entry} standard deviations; need at least {MIN_EVENTS}."
    elif oos and oos["n"] >= 3 and oos["mean"] is not None:
        tr = ts(trades_test)
        if oos["mean"] > 0 and (oos["p"] is not None and oos["p"] < 0.05) and tr["avg_net"] is not None and tr["avg_net"] > 0:
            verdict, why = "converges", "Gaps closed on average in the held-out final 30%, significantly, and the rule made money after costs."
        elif oos["mean"] > 0 and best and best["train"]["p"] is not None and best["train"]["p"] < 0.05:
            verdict, why = "mixed", "Gaps closed on average and the pattern pointed the right way on held-out data, but it was not clearly significant or not profitable after costs."
        elif best and best["train"]["mean"] is not None and best["train"]["mean"] > 0 and oos["mean"] <= 0:
            verdict, why = "did_not_hold", "Gaps closed in the first 70% of history but not in the held-out final 30%."
    chart = {"dates": [str(d.date()) for d in z.index], "z": [round(float(v), 3) for v in z.to_numpy()],
             "spread": [round(float(v), 3) for v in spread.to_numpy()], "split_date": str(z.index[split].date())}
    return {
        "status": "completed", "synthetic": a.startswith("DEMO_"), "a": {"id": a, "label": NAMES.get(a, va.label)}, "b": {"id": b, "label": NAMES.get(b, vb.label)},
        "settings": {"entry_z": entry, "window_days": WINDOW, "lookback_years": years, "days_used": n,
                     "first_date": str(z.index[0].date()), "last_date": str(z.index[-1].date()),
                     "train_share": TRAIN_FRAC, "cost_bps_per_leg": COST_BPS_PER_LEG},
        "events": {"total": len(events), "train": len(train_pos), "test": len(test_pos),
                   "recent": [{"date": str(d.date()), "z": round(float(z.loc[d]), 2),
                               "x_cheap": bool(z.loc[d] < 0)} for d in events[-5:]]},
        "horizons": rows, "control_mean_reversion": control,
        "best_horizon_days": best_h, "out_of_sample": oos,
        "relationship": stat, "trades_all": ts(trades_all), "trades_test": ts(trades_test),
        "current": {"z": round(float(z.iloc[-1]), 2), "spread": round(float(spread.iloc[-1]), 2), "date": str(z.index[-1].date())},
        "verdict": verdict, "verdict_reason": why, "chart": chart,
        "assumptions": ["Equal-dollar legs (hedge ratio 1); no borrow or financing costs; 5 bp per leg per side.",
                        "The gap is measured against the trailing 60-day mean, known at each close.",
                        "Best horizon chosen on the first 70% of days; evaluated once on the last 30%.",
                        "Cointegration test and half-life describe the whole sample and are not used in the verdict.",
                        "Statistical arbitrage is not risk-free: relationships can break permanently."],
    }
