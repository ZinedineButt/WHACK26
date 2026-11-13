"""Second-level research on a pair result: is it luck, is it stable, does it hold on similar pairs, what would it have earned,
and how does it compare with published work. Everything numeric comes from the price data; the AI only writes the comparison
with papers, is limited to papers actually retrieved, and may not introduce numbers that were not computed."""
import json
import re

import numpy as np
import pandas as pd

from . import literature
from . import pairs as P
from .llm import SOURCE_FALLBACK, SOURCE_GEMINI, SOURCE_TEMPLATE, unsupported_numbers
from .strength import label as strength_label

CREDIT = {"pass": 1.0, "inconclusive": 0.5, "fail": 0.0}
WEIGHTS = {"gaps_close": 2, "placebo": 2, "holdout": 4, "stable_years": 1, "threshold": 1, "after_costs": 2, "cointegration": 1, "peers": 1}
RANK = {"low": 0, "moderate": 1, "high": 2}


def _arrays(core):
    idx = pd.to_datetime(core["chart"]["dates"])
    return pd.Series(core["chart"]["spread"], index=idx), pd.Series(core["chart"]["z"], index=idx)


def _rev_all(spread, z, h):
    """-sign(z_t)*(spread_{t+h}-spread_t) for every day t where t+h exists (NaN elsewhere)."""
    s, zz = spread.to_numpy(), z.to_numpy()
    out = np.full(len(s), np.nan)
    out[: len(s) - h] = -np.sign(zz[: len(s) - h]) * (s[h:] - s[: len(s) - h])
    return out


def placebo(core, n_perm=2000, seed=11):
    """Do real divergences close more than ordinary days with a mild gap? Random-day null with the same number of events."""
    spread, z = _arrays(core)
    h = core.get("best_horizon_days") or 5
    rev = _rev_all(spread, z, h)
    entry = core["settings"]["entry_z"]
    ev = [i for i in range(len(z)) if abs(z.iloc[i]) >= entry and (i == 0 or abs(z.iloc[i - 1]) < entry) and not np.isnan(rev[i])]
    pool = [i for i in range(len(z)) if abs(z.iloc[i]) >= 0.5 and not np.isnan(rev[i])]
    if len(ev) < P.MIN_EVENTS or len(pool) < 3 * len(ev):
        return {"horizon_days": h, "events": len(ev), "observed_mean": None, "null_mean": None, "p": None}
    obs = float(np.mean(rev[ev]))
    rng = np.random.default_rng(seed)
    nulls = np.array([np.mean(rev[rng.choice(pool, size=len(ev), replace=False)]) for _ in range(n_perm)])
    return {"horizon_days": h, "events": len(ev), "observed_mean": round(obs, 4), "null_mean": round(float(nulls.mean()), 4),
            "null_95th": round(float(np.percentile(nulls, 95)), 4), "p": round(float((1 + np.sum(nulls >= obs)) / (n_perm + 1)), 4)}


def robustness(core):
    spread, z = _arrays(core)
    h = core.get("best_horizon_days") or 5
    rev = _rev_all(spread, z, h)
    thresholds = []
    for e in (1.5, 2.0, 2.5):
        ev = [i for i in range(len(z)) if abs(z.iloc[i]) >= e and (i == 0 or abs(z.iloc[i - 1]) < e) and not np.isnan(rev[i])]
        thresholds.append({"entry_z": e, "events": len(ev), "mean": round(float(np.mean(rev[ev])), 4) if ev else None,
                           "converged_share": round(float(np.mean(rev[ev] > 0)), 4) if ev else None})
    entry = core["settings"]["entry_z"]
    ev = [i for i in range(len(z)) if abs(z.iloc[i]) >= entry and (i == 0 or abs(z.iloc[i - 1]) < entry) and not np.isnan(rev[i])]
    years = {}
    for i in ev:
        years.setdefault(z.index[i].year, []).append(rev[i])
    by_year = [{"year": y, "events": len(v), "mean": round(float(np.mean(v)), 4)} for y, v in sorted(years.items()) if len(v) >= 3]
    n = core["settings"]["days_used"]
    split = int(n * P.TRAIN_FRAC)
    costs = []
    for bp in (0, 5, 10, 20):
        t_all, t_test = P._trades(spread, z, entry, 0, n, bp), P._trades(spread, z, entry, split, n, bp)
        costs.append({"cost_bps_per_leg": bp, "avg_net_all": round(float(np.mean([t["net"] for t in t_all])), 4) if t_all else None,
                      "avg_net_test": round(float(np.mean([t["net"] for t in t_test])), 4) if t_test else None})
    return {"horizon_days": h, "thresholds": thresholds, "by_year": by_year, "costs": costs}


def peers(svc, core, levels_fn, curated):
    """Run the identical test on other economically linked pairs: does the idea generalise beyond this one pair?"""
    a, b = core["a"]["id"], core["b"]["id"]
    demo = a.startswith("DEMO_")
    fred = lambda x: x.startswith("DCOIL") or x in ("SP500", "NASDAQ100")      # price series that are not company or ETF tickers
    pool = P.DEMO_PAIRS if demo else [p for p in curated if fred(p[0]) == fred(a)]    # compare like with like
    rows = []
    for x, y, _ in pool:
        if {x, y} & {a, b} or len(rows) >= 3:
            continue
        try:
            r = P.run_pair(svc, x, y, core["settings"]["lookback_years"], core["settings"]["entry_z"], levels_fn)
        except Exception:                                   # a comparison pair failing must never sink the main result
            continue
        o = r["out_of_sample"] or {}
        rows.append({"a": r["a"]["label"], "b": r["b"]["label"], "verdict": r["verdict"], "events": r["events"]["total"],
                     "oos_mean": o.get("mean"), "oos_converged_share": o.get("converged_share"), "oos_n": o.get("n"),
                     "half_life_days": r["relationship"]["half_life_days"]})
    ok = [r for r in rows if r["verdict"] != "inconclusive"]
    share = (sum(r["verdict"] in ("converges", "mixed") for r in ok) / len(ok)) if ok else None
    return {"pairs": rows, "share_closing": None if share is None else round(share, 3), "tested": len(ok)}


def simulation(core, capital=10000.0):
    """Pretend-money replay of the frozen rule on the held-out final 30%: capital split equally between the two legs."""
    spread, z = _arrays(core)
    n = core["settings"]["days_used"]
    split = int(n * P.TRAIN_FRAC)
    entry = core["settings"]["entry_z"]
    trades = P._trades(spread, z, entry, split, n)
    if not trades:
        return None
    value, curve, peak, mdd = capital, [{"date": str(spread.index[split].date()), "value": round(capital, 2)}], capital, 0.0
    for t in trades:
        i0 = spread.index.get_loc(pd.Timestamp(t["entry"]))
        exit_date = spread.index[min(i0 + t["days"], n - 1)]
        value *= 1 + 0.5 * t["net"] / 100.0            # half the capital on each leg; net is in % of one leg
        peak = max(peak, value)
        mdd = min(mdd, value / peak - 1)
        curve.append({"date": str(exit_date.date()), "value": round(value, 2)})
    return {"capital": capital, "start": curve[0]["date"], "end": curve[-1]["date"], "final_value": round(value, 2),
            "return_pct": round((value / capital - 1) * 100, 2), "max_drawdown_pct": round(mdd * 100, 2),
            "trades": len(trades), "win_rate": round(float(np.mean([t["net"] > 0 for t in trades])), 3), "curve": curve,
            "note": "Paper trade on held-out data only. Equal-dollar legs, 5 bp per leg per side, no borrowing or financing cost."}


def _res(ok_pass, ok_maybe):
    return "pass" if ok_pass else "inconclusive" if ok_maybe else "fail"


def checks(core, pl, rb, pe):
    h = core["horizons"]
    best = next((r for r in h if r["horizon_days"] == core.get("best_horizon_days")), None)
    allr = best["all"] if best else {}
    oos = core.get("out_of_sample") or {}
    out = []
    out.append({"id": "gaps_close", "name": "Gaps closed on average", "weight": WEIGHTS["gaps_close"],
                "result": _res((allr.get("mean") or 0) > 0 and (allr.get("p") or 1) < 0.05, (allr.get("mean") or 0) > 0) if allr else "fail",
                "detail": f"Best horizon {core.get('best_horizon_days')} day(s): average {allr.get('mean')} points over {allr.get('n')} divergences."})
    out.append({"id": "placebo", "name": "Beats random days (placebo)", "weight": WEIGHTS["placebo"],
                "result": "inconclusive" if pl["p"] is None else _res(pl["p"] < 0.05, pl["p"] < 0.15),
                "detail": "Real divergences versus the same number of randomly chosen mild-gap days (2000 shuffles)." +
                          ("" if pl["p"] is None else f" p = {pl['p']}.")})
    holdout_ok = oos.get("n", 0) >= 3 and oos.get("mean") is not None
    out.append({"id": "holdout", "name": "Held-out final 30% (tested once)", "weight": WEIGHTS["holdout"], "critical": True,
                "result": "inconclusive" if not holdout_ok else _res(oos["mean"] > 0 and (oos.get("p") or 1) < 0.05, oos["mean"] > 0),
                "detail": f"{oos.get('n', 0)} held-out divergences; the horizon was chosen on the first 70% only."})
    yrs = rb["by_year"]
    pos = [y for y in yrs if y["mean"] > 0]
    share = len(pos) / len(yrs) if yrs else None
    out.append({"id": "stable_years", "name": "Works across calendar years", "weight": WEIGHTS["stable_years"],
                "result": "inconclusive" if share is None else _res(share >= 0.6, share >= 0.4),
                "detail": f"{len(pos)} of {len(yrs)} years with at least 3 divergences showed closing." if yrs else "Too few divergences per year."})
    th = [t for t in rb["thresholds"] if t["mean"] is not None]
    npos = sum(t["mean"] > 0 for t in th)
    out.append({"id": "threshold", "name": "Not sensitive to the trigger level", "weight": WEIGHTS["threshold"],
                "result": "inconclusive" if not th else _res(npos == len(th), npos >= 2),
                "detail": f"Closing was positive at {npos} of {len(th)} trigger levels (1.5, 2, 2.5 standard deviations)."})
    ta, tt = core["trades_all"], core["trades_test"]
    pa, pt = (ta.get("avg_net") or 0) > 0, (tt.get("avg_net") or 0) > 0
    out.append({"id": "after_costs", "name": "Profitable after trading costs", "weight": WEIGHTS["after_costs"], "critical": True,
                "result": "inconclusive" if ta.get("n", 0) < 3 else _res(pa and pt, pa or pt),
                "detail": f"Average per trade after costs: all data {ta.get('avg_net')}, held-out {tt.get('avg_net')} (% of one leg)."})
    adf = core["relationship"].get("adf_p")
    out.append({"id": "cointegration", "name": "Gap is statistically mean-reverting", "weight": WEIGHTS["cointegration"],
                "result": "inconclusive" if adf is None else _res(adf < 0.05, adf < 0.15),
                "detail": f"ADF p = {adf} on the whole sample; half-life {core['relationship'].get('half_life_days')} days (descriptive)."})
    sh = pe.get("share_closing")
    out.append({"id": "peers", "name": "Works on similar pairs", "weight": WEIGHTS["peers"],
                "result": "inconclusive" if sh is None else _res(sh >= 0.6, sh >= 0.4),
                "detail": f"{round((sh or 0) * pe['tested'])} of {pe['tested']} comparable pairs also showed closing." if sh is not None else "No comparable pairs could be tested."})
    return out


def strength(chk):
    ran = [c for c in chk if c["result"] in CREDIT]
    pct = round(100 * sum(c["weight"] * CREDIT[c["result"]] for c in ran) / sum(c["weight"] for c in ran)) if ran else None
    return {"pct": pct, "label": strength_label(pct), "passed": sum(c["result"] == "pass" for c in ran), "ran": len(ran)}


def deepen(svc, core, levels_fn, curated):
    pl, rb = placebo(core), robustness(core)
    pe = peers(svc, core, levels_fn, curated)
    chk = checks(core, pl, rb, pe)
    st = strength(chk)
    holdout = next(c for c in chk if c["id"] == "holdout")["result"]
    costs = next(c for c in chk if c["id"] == "after_costs")["result"]
    # the headline verdict now follows the graded checks, not just the held-out mean
    if core["verdict"] == "inconclusive":
        pass
    elif st["pct"] is not None and st["pct"] >= 70 and holdout == "pass" and costs != "fail":
        core["verdict"], core["verdict_reason"] = "converges", "Gaps closed, beat random days, held up on the locked-away final 30% and stayed profitable after costs."
    elif holdout == "fail" or (st["pct"] is not None and st["pct"] < 40):
        core["verdict"], core["verdict_reason"] = ("did_not_hold" if core["verdict"] in ("mixed", "did_not_hold") else "no_evidence",
                                                   "The checks do not support a reliable pull-back after a divergence.")
    elif core["verdict"] == "converges":
        core["verdict"], core["verdict_reason"] = "mixed", "Some checks support closing, but not enough of them to call it reliable."
    core.update({"placebo": pl, "robustness": rb, "peer_pairs": pe, "checks": chk, "strength": st,
                 "simulation": simulation(core) if core["verdict"] in ("converges", "mixed") and (core["trades_test"].get("avg_net") or 0) > 0 else None})
    return core


# ------------------------------------------------------------------ literature + AI comparison
from typing import List, Literal  # noqa: E402

from pydantic import BaseModel  # noqa: E402


class _PaperNote(BaseModel):
    paper_index: int
    stance: Literal["supports", "mixed", "contradicts", "not_relevant"]
    note: str


class _Point(BaseModel):
    severity: Literal["minor", "moderate", "major"]
    point: str


class _Critique(BaseModel):
    confidence: Literal["low", "moderate", "high"]
    comparison: str
    critique: List[_Point]
    paper_assessments: List[_PaperNote]


def _facts(core):
    keep = {"assets": [core["a"]["label"], core["b"]["label"]], "settings": core["settings"], "events": {"total": core["events"]["total"]},
            "best_horizon_days": core["best_horizon_days"],
            "horizons": [{"h": r["horizon_days"], "all_mean": r["all"]["mean"], "all_closed_share": r["all"]["converged_share"],
                          "test_mean": r["test"]["mean"], "test_n": r["test"]["n"]} for r in core["horizons"]],
            "relationship": core["relationship"], "placebo": core["placebo"], "robustness": core["robustness"],
            "trades_all": core["trades_all"], "trades_test": core["trades_test"], "peer_pairs": core["peer_pairs"],
            "strength": core["strength"], "verdict": core["verdict"], "checks": [{k: c[k] for k in ("id", "result", "detail")} for c in core["checks"]]}
    return json.loads(json.dumps(keep, default=str))


def search_papers(core):
    a, b = core["a"]["label"], core["b"]["label"]
    for q in (f"pairs trading statistical arbitrage {a} {b} cointegration", "pairs trading statistical arbitrage mean reversion cointegration"):
        res = literature.search(None, limit=literature.MAX_PAPERS, query=q)
        if res["status"] == "completed":
            return res
    return res


def template_critique(core, papers, source):
    c = {x["id"]: x["result"] for x in core["checks"]}
    pts = []
    if core["events"]["total"] < 30:
        pts.append({"severity": "major", "point": "There were few divergences, so averages are noisy and a few lucky cases could drive the result."})
    if c.get("after_costs") != "pass":
        pts.append({"severity": "major", "point": "The gap may close on average yet not by enough to pay trading costs once borrowing and slippage are included."})
    if c.get("holdout") != "pass":
        pts.append({"severity": "major", "point": "The pattern was not clearly confirmed on the final 30% of data that was kept locked away."})
    pts.append({"severity": "moderate", "point": "Relationships between assets can break permanently (a merger, a product shift, a regime change); a past pull-back is no guarantee."})
    pts.append({"severity": "minor", "point": "Equal-dollar legs and a 60-day trailing mean are simple choices; other hedge ratios could change the result."})
    cap = "high" if core["strength"]["pct"] and core["strength"]["pct"] >= 75 else "moderate" if core["strength"]["pct"] and core["strength"]["pct"] >= 50 else "low"
    return {"comparison": "Published studies of pairs trading generally find that gaps between closely related assets can narrow, but that profits shrink after costs "
                          "and as the strategy becomes crowded. Read the retrieved papers below for the details; this summary is a template, not an AI reading of them.",
            "critique": pts, "paper_assessments": [], "confidence": cap, "source": source}


def critique_with_papers(reasoner, core, papers):
    facts = _facts(core)
    pct = core["strength"]["pct"] or 0
    cap = "high" if pct >= 75 else "moderate" if pct >= 50 else "low"
    if not reasoner.configured:
        return template_critique(core, papers, SOURCE_TEMPLATE)
    prompt = (
        "You are a sceptical quantitative researcher. Compare the automated pairs-trading finding below with the retrieved research papers. "
        "Using ONLY the JSON facts and the papers: (1) write `comparison`, at most 90 words, saying how this finding agrees or disagrees with what "
        "the papers report about pairs trading / statistical arbitrage (convergence, profits after costs, decay over time, cointegration), citing papers "
        "only as [index]; (2) list 3 to 5 strongest reasons the result may be wrong or not tradeable; (3) for each paper say whether it supports, "
        "contradicts, is mixed on, or is not relevant to this finding, in at most 25 words. Do not introduce numbers that are not in the facts. "
        "Never say anything is proven. Rate confidence honestly.\nFACTS: " + json.dumps(facts) +
        "\nPAPERS: " + json.dumps([{"index": p["index"], "title": p["title"], "year": p["year"], "abstract": (p["abstract"] or "")[:450]} for p in papers]))
    try:
        data = _Critique.model_validate(reasoner._call(prompt, _Critique)).model_dump()
        idxs = {p["index"] for p in papers}
        pa = [{"paper_index": a["paper_index"], "stance": a["stance"], "note": a["note"][:300]} for a in data["paper_assessments"] if a["paper_index"] in idxs]
        pts = [{"severity": c["severity"], "point": c["point"][:400]} for c in data["critique"]][:6]
        text = " ".join([data["comparison"], *[p["point"] for p in pts], *[a["note"] for a in pa]])
        bad = unsupported_numbers(text, facts)
        if bad or re.search(r"\bprov(en|es|e)\b", text, re.I) or not pts:
            raise ValueError(f"unsupported content {bad[:3]}")
        conf = data["confidence"] if RANK[data["confidence"]] <= RANK[cap] else cap
        return {"comparison": data["comparison"][:700], "critique": pts, "paper_assessments": pa, "confidence": conf, "confidence_cap": cap,
                "source": SOURCE_GEMINI}
    except Exception:
        return template_critique(core, papers, SOURCE_FALLBACK)


def literature_and_critique(reasoner, core):
    lit = search_papers(core)
    papers = [{"index": p["index"], "title": p["title"], "year": p["year"], "venue": p["venue"], "url": p["url"],
               "cited_by": p["cited_by"], "abstract": p["abstract"]} for p in lit["papers"]]
    crit = critique_with_papers(reasoner, core, papers)
    shown = [{k: p[k] for k in ("index", "title", "year", "venue", "url", "cited_by")} for p in papers]
    return {"literature": {"status": lit["status"], "provider": lit["provider"], "query": lit["query"], "papers": shown, "note": lit["note"]},
            "critique": crit}
