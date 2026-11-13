"""/api/research/pairs: test whether two assets that diverge come back together, and let the AI suggest candidate pairs."""
from typing import List

from flask import jsonify, request
from pydantic import BaseModel

from . import registry as R
from .api import _state, research_bp
import threading
import time

from . import pairs_deep
from .pairs import DEMO_PAIRS, NAMES, PairError, run_pair

CURATED = [
    ("KO", "PEP", "Two rival soft-drink giants with near-identical customers and costs."),
    ("V", "MA", "The two dominant card networks earn money from the same payment volumes."),
    ("XOM", "CVX", "Both are integrated US oil majors exposed to the same oil price."),
    ("GLD", "SLV", "Gold and silver both act as stores of value and often move together."),
    ("DCOILWTICO", "DCOILBRENTEU", "US and European crude oil benchmarks, linked by shipping and refining arbitrage."),
    ("HD", "LOW", "The two big US home-improvement retailers sell to the same housing market."),
    ("JPM", "BAC", "Two of the largest US banks with similar interest-rate exposure."),
]
POOL = ["KO", "PEP", "V", "MA", "XOM", "CVX", "GLD", "SLV", "HD", "LOW", "JPM", "BAC", "WMT", "TGT", "DAL", "UAL", "SPY", "QQQ",
        "NVDA", "AMD", "AAPL", "MSFT", "GOOGL", "META", "UPS", "FDX", "MCD", "SBUX", "DCOILWTICO", "DCOILBRENTEU", "SP500", "NASDAQ100"]


class _Suggest(BaseModel):
    class P(BaseModel):
        a: str
        b: str
        reason: str
    pairs: List[P]


def _describe(a, b):
    return {"a": a, "b": b, "a_name": NAMES.get(a, a), "b_name": NAMES.get(b, b)}


_core_cache, _core_lock = {}, threading.Lock()


def _core(svc, a, b, years, entry, levels_fn):
    """Full pair study (core + second-level research), cached 30 minutes so the literature step can reuse it."""
    key = (a, b, years, entry)
    with _core_lock:
        hit = _core_cache.get(key)
        if hit and time.time() - hit[0] < 1800:
            return hit[1]
    res = pairs_deep.deepen(svc, run_pair(svc, a, b, years, entry, levels_fn), levels_fn, CURATED)
    with _core_lock:
        _core_cache[key] = (time.time(), res)
    return res


@research_bp.post("/api/research/pairs/literature")
def pairs_literature():
    """Papers on pairs trading plus the AI's comparison with this pair's results (second request, so the numbers show first)."""
    from .globe_api import _levels_cached
    body = request.get_json(silent=True) or {}
    try:
        st = _state()
        core = _core(st["services"], str(body.get("a") or "").strip().upper(), str(body.get("b") or "").strip().upper(),
                     float(body.get("lookback_years", 3)), float(body.get("entry_z", 2.0)), _levels_cached)
        return jsonify(pairs_deep.literature_and_critique(st["reasoner"], core))
    except PairError as e:
        return jsonify({"error": e.message}), e.status
    except (TypeError, ValueError):
        return jsonify({"error": "lookback_years and entry_z must be numbers"}), 400
    except Exception as e:
        return jsonify({"error": f"Literature step failed ({type(e).__name__})"}), 500


@research_bp.post("/api/research/pairs")
def pairs_run():
    from .globe_api import _levels_cached          # imported here: globe_api and pairs_api both import api at module load
    body = request.get_json(silent=True) or {}
    try:
        res = _core(_state()["services"], str(body.get("a") or "").strip().upper(), str(body.get("b") or "").strip().upper(),
                    float(body.get("lookback_years", 3)), float(body.get("entry_z", 2.0)), _levels_cached)
    except PairError as e:
        return jsonify({"error": e.message}), e.status
    except (TypeError, ValueError):
        return jsonify({"error": "lookback_years and entry_z must be numbers"}), 400
    except Exception as e:                                           # always answer with JSON the page can show
        return jsonify({"error": f"The pair test failed ({type(e).__name__}: {str(e)[:120]})"}), 500
    return jsonify(res)


@research_bp.get("/api/research/pairs/suggest")
def pairs_suggest():
    st = _state()
    reasoner = st["reasoner"]
    out, source = None, "curated"
    if reasoner.configured:
        try:
            names = "\n".join(f"{k}: {NAMES.get(k, k)}" for k in POOL)
            prompt = ("Suggest 5 pairs of assets whose prices are economically linked (same industry, substitutes, or the same "
                      "underlying commodity) so that if they drift apart they might converge again (statistical arbitrage). "
                      "Use ONLY ids from this list, vary the asset types, and give each pair a reason of at most 20 words with no numbers.\n" + names)
            raw = reasoner._call(prompt, _Suggest)
            seen, out = set(), []
            for p in raw["pairs"]:
                a, b = p["a"].strip().upper(), p["b"].strip().upper()
                if a in POOL and b in POOL and a != b and frozenset((a, b)) not in seen:
                    seen.add(frozenset((a, b)))
                    out.append({**_describe(a, b), "reason": p["reason"][:160]})
            source = "gemini" if out else "curated"
        except Exception:
            out = None
    if not out:
        out = [{**_describe(a, b), "reason": why} for a, b, why in CURATED[:5]]
    return jsonify({"source": source, "pairs": out[:5],
                    "demo": [{**_describe(a, b), "reason": why} for a, b, why in DEMO_PAIRS[:2]]})
