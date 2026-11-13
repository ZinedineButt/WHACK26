"""Globe endpoints: one company's daily-return correlation with the other tracked companies, and the top papers about it.

Every link is PREDICTIVE and free of look-ahead: the leader's return on day t is paired with the follower's return on day t+1
(LAG = 1), never the same day. Returns are split/dividend-adjusted daily log returns from the same DataService as the research
engine, with Newey-West p-values. Nothing is simulated: if no data provider works the endpoint says so."""
import datetime as dt
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from typing import List, Literal

import numpy as np
import pandas as pd
from flask import jsonify, request
from pydantic import BaseModel

from . import literature
from . import registry as R
from . import stats as S
from . import transforms as T
from .api import _state, research_bp
from .providers import ProviderError

DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD"
WINDOWS = {"3m": 0.25, "1y": 1, "3y": 3}
_cache, _lock = {}, threading.Lock()
CACHE_S = 600
MIN_OBS = 40
LAG = 1                  # leader on day t vs follower on day t+1: no look-ahead
MAX_NET_LINKS = 250
COMPANY = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOGL": "Alphabet",
           "META": "Meta Platforms", "TSLA": "Tesla", "AMD": "Advanced Micro Devices"}


MAX_SYMBOLS = 80
FETCH_WORKERS = 4


def _symbols():
    """Symbols to analyse: the `symbols` query parameter (the companies on the map), else the market stream's env list.
    Anything that is not a syntactically valid ticker is dropped."""
    raw = request.args.get("symbols") or os.environ.get("ALPACA_SYMBOLS", DEFAULT_SYMBOLS)
    out = []
    for s in raw.split(","):
        s = s.strip().upper()
        if R.TICKER_RE.match(s) and s not in out:
            out.append(s)
    return out[:MAX_SYMBOLS]


def _valid_symbol(sym):
    return bool(R.TICKER_RE.match(sym or ""))


def _cached(key, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_S:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
    return val


LEVEL_TTL = 6 * 3600          # same as the provider cache
FAIL_TTL = 120                # a failed symbol is not retried for 2 minutes (avoids retry storms on rate limits)
BLOCK_S = 600                 # after an HTTP 429 from the primary provider, go straight to Alpaca for 10 minutes
_levels_cache, _blocked = {}, {"until": 0.0}


def _alpaca_levels(svc, v, years):
    alpaca = getattr(svc.data, "alpaca", None)
    if v.source != "equity" or alpaca is None or not alpaca.configured:
        return None
    end = dt.date.today()
    return alpaca.fetch(v, end - dt.timedelta(days=int(365.25 * years)), end)[0]


def _levels(svc, v, years):
    """Price levels from the primary equity provider; if it is rate-limited (free tiers return HTTP 429 after a few dozen
    symbols) fall back to Alpaca daily bars, and keep using Alpaca for a while instead of waiting out every retry."""
    if time.time() < _blocked["until"]:
        try:
            lv = _alpaca_levels(svc, v, years)
            if lv is not None:
                return lv
        except ProviderError:
            pass
    try:
        return svc.data.get(v, years)[0]
    except ProviderError as first:
        if "429" in str(first):
            _blocked["until"] = time.time() + BLOCK_S
        try:
            lv = _alpaca_levels(svc, v, years)
        except ProviderError:
            lv = None
        if lv is None:
            raise first
        return lv


def _levels_cached(svc, v, years):
    """Levels cached in memory per (symbol, years): successes for LEVEL_TTL, failures for FAIL_TTL. Raises ProviderError."""
    key = (v.id, years)
    with _lock:
        hit = _levels_cache.get(key)
    if hit:
        ts, lv, err = hit
        if lv is not None and time.time() - ts < LEVEL_TTL:
            return lv
        if lv is None and time.time() - ts < FAIL_TTL:
            raise ProviderError(err)
    try:
        lv = _levels(svc, v, years)
    except ProviderError as e:
        with _lock:
            _levels_cache[key] = (time.time(), None, str(e))
        raise
    with _lock:
        _levels_cache[key] = (time.time(), lv, None)
    return lv


def _symbol_returns(svc, symbol, window):
    """(daily log returns trimmed to the window, None) or (None, reason). Successes are cached per symbol so every view
    reuses them; failures are not cached, so a rate-limited symbol is retried on the next request."""
    key = ("ret", symbol, window)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_S:
            return hit[1]
    years = WINDOWS[window]
    v, why = svc.registry.resolve(symbol)
    if v is None:
        return None, why
    try:
        level = _levels_cached(svc, v, max(years, 1))   # fetch >= 1y so a 3-month window still has history
    except ProviderError as e:
        return None, str(e)
    r = T.apply(level, R.LOG_RETURN).dropna()
    if years < 1:
        r = r[r.index >= r.index[-1] - pd.Timedelta(days=int(365.25 * years))]
    with _lock:
        _cache[key] = (time.time(), (r, None))
    return r, None


def _returns(svc, window, symbols):
    """Daily log-return series per symbol, fetched concurrently. Returns (rets, errors)."""
    rets, errors = {}, {}
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for s, (r, why) in zip(symbols, pool.map(lambda x: _symbol_returns(svc, x, window), symbols)):
            if r is None:
                errors[s] = why
            else:
                rets[s] = r
    return rets, errors


def _pair(a, b, n_links, lag=LAG):
    """Correlation of a's return on day t with b's return on day t+lag (a leads b). Returns None if too little data."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    x = df.iloc[:, 0].shift(lag)
    d = pd.concat([x, df.iloc[:, 1]], axis=1).dropna()
    if len(d) < MIN_OBS:
        return None
    xv, yv = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    if np.std(xv) == 0 or np.std(yv) == 0:
        return None
    rv, p = S.hac_corr(xv, yv)
    return {"r": round(rv, 4), "p_value": round(p, 5), "n": int(len(d)),
            "significant": bool(p < 0.05 / max(1, n_links))}          # Bonferroni across the links shown


def _lag_matrix(rets, lag=LAG):
    """M[i][j] = corr(return_i on day t, return_j on day t+lag), pairwise-complete. Fast numpy-free pandas pass for ranking."""
    df = pd.DataFrame(rets)
    cols = list(df.columns)
    rows = {}
    for i in cols:
        rows[i] = df.corrwith(df[i].shift(lag))               # corr(i_{t-lag}, j_t) for every follower j
    return pd.DataFrame(rows).T.loc[cols, cols] if rows else pd.DataFrame()


def correlations(svc, symbol, window, symbols):
    rets, errors = _returns(svc, window, symbols)
    if symbol not in rets:
        return {"status": "unavailable", "symbol": symbol, "window": window, "links": [],
                "note": errors.get(symbol) or f"No data for {symbol}."}
    links = []
    for s, r in rets.items():
        if s != symbol:
            res = _pair(rets[symbol], r, len(rets) - 1)
            if res:
                links.append({"symbol": s, **res})
    links.sort(key=lambda l: -abs(l["r"]))
    return {"status": "completed" if links else "unavailable", "symbol": symbol, "window": window, "links": links,
            "failed": errors, "lag_days": LAG,
            "note": f"{symbol}'s return today vs each company's return the next trading day. Newey-West p-values, Bonferroni-adjusted."
            if links else "Not enough overlapping data."}


def network(svc, window, symbols):
    """Every pair of tracked companies in its stronger direction (leader -> follower), strongest first (top MAX_NET_LINKS)."""
    rets, errors = _returns(svc, window, symbols)
    syms = sorted(rets)
    cand = []
    if len(syms) > 1:
        m = _lag_matrix({s: rets[s] for s in syms})
        for i, a in enumerate(syms):
            for b in syms[i + 1:]:
                ab, ba = m.loc[a, b], m.loc[b, a]
                ab = 0.0 if pd.isna(ab) else float(ab)
                ba = 0.0 if pd.isna(ba) else float(ba)
                cand.append((a, b) if abs(ab) >= abs(ba) else (b, a))
                cand[-1] = (*cand[-1], max(abs(ab), abs(ba)))
    cand.sort(key=lambda c: -c[2])
    n_pairs = max(1, len(syms) * (len(syms) - 1) // 2)
    links = []
    for lead, follow, _ in cand[:MAX_NET_LINKS]:                 # exact (Newey-West) numbers only for the strongest
        res = _pair(rets[lead], rets[follow], n_pairs)
        if res:
            links.append({"a": lead, "b": follow, **res})
    links.sort(key=lambda l: -abs(l["r"]))
    return {"status": "completed" if links else "unavailable", "window": window, "links": links, "failed": errors,
            "symbols_used": sorted(rets), "lag_days": LAG,
            "note": "Leader's return today vs follower's return the next trading day, for every pair (stronger direction shown)."
            if links else "Not enough overlapping data."}


@research_bp.get("/api/research/globe/correlations")
def globe_correlations():
    symbol = (request.args.get("symbol") or "").strip().upper()
    window = (request.args.get("window") or "1y").lower()
    if not _valid_symbol(symbol):
        return jsonify({"error": f"invalid symbol '{symbol}'"}), 400
    if window not in WINDOWS:
        return jsonify({"error": f"window must be one of {sorted(WINDOWS)}"}), 400
    symbols = _symbols()
    if symbol not in symbols:
        symbols = [symbol] + symbols[:MAX_SYMBOLS - 1]
    svc = _state()["services"]
    try:
        return jsonify(_cached(("corr", symbol, window, tuple(sorted(symbols))), lambda: correlations(svc, symbol, window, symbols)))
    except Exception as e:                                           # never take the globe down
        return jsonify({"status": "unavailable", "symbol": symbol, "window": window, "links": [],
                        "note": f"{type(e).__name__}: {str(e)[:120]}"})


@research_bp.get("/api/research/globe/papers")
def globe_papers():
    symbol = (request.args.get("symbol") or "").strip().upper()
    if not _valid_symbol(symbol):
        return jsonify({"error": f"invalid symbol '{symbol}'"}), 400
    name = literature.clean_query(request.args.get("name") or "")[:60] or COMPANY.get(symbol, symbol)
    if name.upper() == symbol:
        name = f"{symbol} company"
    res = _cached(("papers", symbol, name), lambda: literature.search(None, limit=literature.MAX_PAPERS,
                                                                query=f"{name} stock returns"))
    papers = [{k: p.get(k) for k in ("title", "year", "venue", "url", "cited_by", "source")} for p in res["papers"]]
    return jsonify({"status": res["status"], "symbol": symbol, "query": res["query"], "provider": res["provider"],
                    "papers": papers, "note": res["note"]})


@research_bp.get("/api/research/globe/network")
def globe_network():
    window = (request.args.get("window") or "1y").lower()
    if window not in WINDOWS:
        return jsonify({"error": f"window must be one of {sorted(WINDOWS)}"}), 400
    svc = _state()["services"]
    try:
        symbols = _symbols()
        return jsonify(_cached(("net", window, tuple(sorted(symbols))), lambda: network(svc, window, symbols)))
    except Exception as e:
        return jsonify({"status": "unavailable", "window": window, "links": [], "note": f"{type(e).__name__}: {str(e)[:120]}"})


# ------------------------------------------------------------------ "Ask the globe"
class _AskPair(BaseModel):
    a: str
    b: str
    expected: Literal["positive", "negative", "unclear"]
    reason: str


class _AskOut(BaseModel):
    kind: Literal["lead_lag", "pair_divergence"]
    pairs: List[_AskPair]
    research_question: str


ASK_PROMPT = """You help a user explore stock relationships on a world map. The user asked a question about the market.
Choose at most 4 pairs of companies, using ONLY tickers from the allowed list, that best test the question.
Order each pair as (leader, follower): the company whose move today you think helps predict the other's move the NEXT trading day.
Give the direction you expect (positive = the follower tends to move the same way the next day, negative = the opposite way,
unclear) and a reason of at most 25 words. Do NOT state any numbers or statistics. Also write one clear research question
(max 30 words) that could be tested statistically on daily returns for the best pair.
Set kind = "pair_divergence" if the user asks whether two assets that drift apart / diverge come back together / converge /
mean-revert (pairs trading, statistical arbitrage, spread), otherwise kind = "lead_lag". For pair_divergence pick pairs of closely
related companies (same industry or substitutes) and order does not matter.
Allowed companies (ticker: name):
{allowed}
User question: {question}"""


def _ask_fallback(question, names):
    """No Gemini: pair up the companies the question mentions by ticker or name. Never invents a reason."""
    q = f" {question.lower()} "
    hit = [s for s, n in names.items()
           if f" {s.lower()} " in q.replace("?", " ").replace(",", " ") or (len(n) > 3 and n.lower() in q)]
    hit = hit[:4]
    pairs = [{"a": a, "b": b, "expected": "unclear", "reason": ""} for i, a in enumerate(hit) for b in hit[i + 1:]][:4]
    kind = "pair_divergence" if re.search(r"diverg|converge|come back|pairs? trad|arbitrage|spread|mean.?revert|snap back", question, re.I) else "lead_lag"
    return {"kind": kind, "pairs": pairs, "research_question": ""}, "keyword_match"


def _strength_word(r):
    a = abs(r)
    return "very strong" if a >= 0.6 else "strong" if a >= 0.4 else "moderate" if a >= 0.2 else "weak" if a >= 0.1 else "negligible"


@research_bp.post("/api/research/globe/ask")
def globe_ask():
    body = request.get_json(silent=True) or {}
    question = literature.clean_query(str(body.get("question") or ""))[:200]
    window = str(body.get("window") or "1y").lower()
    if len(question) < 4:
        return jsonify({"error": "ask a question of at least a few words"}), 400
    if window not in WINDOWS:
        return jsonify({"error": f"window must be one of {sorted(WINDOWS)}"}), 400
    names = {}
    for item in (body.get("companies") or [])[:MAX_SYMBOLS]:
        sym = str((item or {}).get("symbol") or "").upper()
        if R.TICKER_RE.match(sym):
            names[sym] = literature.clean_query(str((item or {}).get("company") or sym))[:60]
    if len(names) < 2:
        return jsonify({"error": "no companies supplied"}), 400
    st = _state()
    reasoner, svc = st["reasoner"], st["services"]
    out, source = None, None
    if reasoner.configured:
        try:
            allowed = "\n".join(f"{s}: {n}" for s, n in names.items())
            out = _AskOut(**reasoner._call(ASK_PROMPT.format(allowed=allowed, question=question), _AskOut)).model_dump()
            source = "gemini"
        except Exception:
            out = None
    if out is None:
        out, source = _ask_fallback(question, names)
    pairs, seen = [], set()
    for p in out["pairs"]:
        a, b = p["a"].upper(), p["b"].upper()
        key = frozenset((a, b))
        if a == b or a not in names or b not in names or key in seen:
            continue                                                 # the model may only use tickers on the map
        seen.add(key)
        pairs.append({**p, "a": a, "b": b})
        if len(pairs) == 4:
            break
    if out.get("kind") == "pair_divergence":
        from . import pairs as PR
        studies = []
        for p in pairs:
            try:
                r = PR.run_pair(svc, p["a"], p["b"], 3, 2.0, _levels_cached)
            except PR.PairError:
                continue
            oos = r["out_of_sample"] or {}
            studies.append({**p, "kind": "pair_divergence", "events": r["events"]["total"], "best_horizon_days": r["best_horizon_days"],
                            "oos_n": oos.get("n"), "oos_converged_share": oos.get("converged_share"), "oos_mean": oos.get("mean"),
                            "half_life_days": r["relationship"]["half_life_days"], "adf_p": r["relationship"]["adf_p"],
                            "verdict": r["verdict"], "verdict_reason": r["verdict_reason"], "current_z": r["current"]["z"],
                            "reason": (p["reason"] or "")[:200]})
        return jsonify({"status": "completed" if studies else "no_results", "source": source, "kind": "pair_divergence",
                        "question": question, "window": "3y", "pairs": studies,
                        "research_question": (out.get("research_question") or "")[:300],
                        "note": None if studies else "Could not test those pairs (no data, or not enough overlapping history)."})
    rets, errors = _returns(svc, window, sorted({s for p in pairs for s in (p["a"], p["b"])}))
    results = []
    for p in pairs:
        if p["a"] in rets and p["b"] in rets:
            fwd, back = _pair(rets[p["a"]], rets[p["b"]], len(pairs)), _pair(rets[p["b"]], rets[p["a"]], len(pairs))
            options = [(p["a"], p["b"], r) for r in [fwd] if r] + [(p["b"], p["a"], r) for r in [back] if r]
            if options:
                lead, follow, res = max(options, key=lambda o: abs(o[2]["r"]))   # the direction with the stronger next-day link
                p = {**p, "a": lead, "b": follow, "swapped": lead != p["a"]}
                sign = "positive" if res["r"] > 0 else "negative"
                results.append({**p, **res, "strength": _strength_word(res["r"]), "direction": sign, "lag_days": LAG,
                                "matches_expectation": None if p["expected"] == "unclear" else p["expected"] == sign,
                                "reason": (p["reason"] or "")[:200]})
    note = None if results else ("Could not find two companies on the map in that question. Try naming them, e.g. "
                                 "\"does Walmart move with Coca-Cola?\"." if source == "keyword_match" else
                                 "No data available for the suggested pairs: " + "; ".join(f"{k}: {v[:50]}" for k, v in errors.items()))
    return jsonify({"status": "completed" if results else "no_results", "source": source, "kind": "lead_lag", "question": question, "window": window,
                    "pairs": results, "research_question": (out.get("research_question") or "")[:300], "note": note})


# ------------------------------------------------------------------ replay of the past month
REPLAY_DAYS = 21
REPLAY_PEERS = 3


def replay(svc, symbols, days):
    """Real daily % moves for the last `days`+1 trading days. Each company's peers are the companies whose NEXT-day move its
    move best predicts, measured on data that ends BEFORE the replayed month (so the replay is out of sample: no look-ahead).
    The globe judges each day's leaders against the following day's real peer moves. Nothing is simulated."""
    rets1y, errors = _returns(svc, "1y", symbols)
    if not rets1y:
        return {"status": "unavailable", "days": [], "moves": {}, "peers": {}, "failed": errors,
                "note": "No market data available for the replay."}
    ref = max(rets1y.values(), key=len)
    dates = list(ref.index[-(days + 1):])                      # +1: the last day is only ever an outcome
    moves = {s: [None if d not in r.index else round(float(r.loc[d]), 3) for d in dates] for s, r in rets1y.items()}
    pct = {s: [None if v is None else round((math.exp(v / 100) - 1) * 100, 3) for v in series] for s, series in moves.items()}
    close = {}
    for s_ in moves:                                                   # real adjusted closes on the same dates (for zoomed-in views)
        v, _ = svc.registry.resolve(s_)
        try:
            lv = _levels_cached(svc, v, 1) if v is not None else None
        except ProviderError:
            lv = None
        close[s_] = [None if lv is None or d not in lv.index else round(float(lv.loc[d]), 2) for d in dates]
    prior = {s: r[r.index < dates[0]] for s, r in rets1y.items()}
    peers = {}
    if len(prior) > 1:
        m = _lag_matrix(prior)
        for s in m.index:
            row = m.loc[s].drop(s).dropna()
            top = row.reindex(row.abs().sort_values(ascending=False).index)[:REPLAY_PEERS]
            peers[s] = [{"symbol": p, "r": round(float(v), 4)} for p, v in top.items()]
    return {"status": "completed", "days": [d.date().isoformat() for d in dates], "moves": moves, "pct": pct, "close": close, "peers": peers,
            "failed": errors, "lag_days": LAG,
            "note": "Peers = companies whose next-day move this company's move best predicted over the year before the replayed month."}


@research_bp.get("/api/research/globe/replay")
def globe_replay():
    try:
        days = max(5, min(30, int(request.args.get("days", REPLAY_DAYS))))
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400
    svc = _state()["services"]
    symbols = _symbols()
    try:
        return jsonify(_cached(("replay", days, tuple(sorted(symbols))), lambda: replay(svc, symbols, days)))
    except Exception as e:
        return jsonify({"status": "unavailable", "days": [], "moves": {}, "peers": {}, "note": f"{type(e).__name__}: {str(e)[:120]}"})
