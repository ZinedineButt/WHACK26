"""Flask blueprint: /api/research/* (+ /api/health when the host app has none)."""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from flask import Blueprint, current_app, jsonify, request

from . import registry as R
from .engine import MAX_HYPOTHESES_CAP, new_report as new_report_v2, run_autonomous
from .llm import Reasoner
from .pipeline import new_report, run_pipeline
from .presets import PRESETS, public_presets
from .providers import DataService, ProviderError
from .simulate import MAX_CAPITAL, MIN_CAPITAL, PERIODS, SimulationError, benchmark_variable_id, run_simulation
from .store import to_jsonable
from .store import Store

research_bp = Blueprint("research", __name__)

MODES = ("autonomous", "guided", "demo_autonomous")
LOOKBACKS = (1, 3, 5)
MAX_ACTIVE_RUNS = 3
MAX_BODY_BYTES = 20_000
DEFAULT_THEME = "What drives moves in the stock market?"
EXAMPLE_THEMES = ["Oil, natural gas and US equity indices", "Market volatility and technology stocks",
                  "The US dollar and global equities", "Credit stress and equity drawdowns"]
ALLOWED_KEYS = {"mode", "theme", "question", "lookback_years", "max_hypotheses", "universe"}


@dataclass
class Services:
    reasoner: Reasoner
    offline_reasoner: Reasoner
    data: DataService

    @property
    def registry(self):
        ok, why, _ = self.data.equity_status()
        return R.Registry(fred_available=True, equity_available=ok, equity_reason=None if ok else why)


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def _state():
    return current_app.extensions["research"]


def init_research(app, db_path, sync=None, reasoner=None, data=None, max_workers=2, timeout_s=600):
    """sync=True runs jobs inline (tests). Default: bounded thread pool.
    Single-process only; runs interrupted by a restart are marked failed on next start."""
    # Research-only secrets (Gemini/FRED) live in backend/.env.local (git-ignored via .env.*) so the
    # tracked backend/.env that the market map uses is never modified.
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(db_path).resolve().parent.parent / ".env.local")
    if sync is None:
        sync = os.environ.get("RESEARCH_SYNC", "").lower() in ("1", "true")
    Path(db_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    store = Store(db_path)
    store.fail_interrupted()
    reasoner = reasoner or Reasoner()
    app.extensions["research"] = {
        "store": store, "sync": sync, "reasoner": reasoner,
        "services": Services(reasoner, Reasoner(api_key=""), data or DataService(cache_path=db_path)),
        "pool": None if sync else ThreadPoolExecutor(max_workers=max_workers),
        "timeout_s": int(os.environ.get("RESEARCH_JOB_TIMEOUT", timeout_s)),
    }
    app.register_blueprint(research_bp)
    if not any(r.rule == "/api/health" for r in app.url_map.iter_rules()):
        app.add_url_rule("/api/health", "research_health", health)


def health():
    return jsonify(_health_body())


def _health_body():
    st = _state()
    d = st["services"].data
    return {"status": "ok", "gemini_configured": st["reasoner"].configured,
            "alpaca_configured": d.alpaca.configured, "alpaca_usable": d.alpaca.check()[0],
            "tiingo_configured": d.tiingo.configured, "equity_data_usable": d.equity_status()[0], "equity_data_status": d.equity_status()[1],
            "fred_mode": d.fred.mode}


@research_bp.get("/api/research/health")
def research_health():
    return jsonify(_health_body())


# ------------------------------------------------------------------ catalog / presets
@research_bp.get("/api/research/presets")
def presets():
    return jsonify({"presets": public_presets(), "note": "Legacy synthetic presets (mode: demo). "
                    "POST {\"preset_id\": ...} still works; prefer mode=autonomous|guided|demo_autonomous."})


@research_bp.get("/api/research/catalog")
def catalog():
    st = _state()
    svc = st["services"]
    reg = svc.registry
    return jsonify({
        "providers": {
            "gemini": {"configured": st["reasoner"].configured, "model": st["reasoner"].model if st["reasoner"].configured else None},
            "fred": {"configured": True, "mode": svc.data.fred.mode,
                     "note": "Without FRED_API_KEY the public CSV endpoint is used (same series, no vintages)."},
            "equity_data": {"usable": svc.data.equity_status()[0], "status": svc.data.equity_status()[1],
                            "provider": (svc.data.equity_status()[2].name if svc.data.equity_status()[2] else None),
                            "note": "Stock/ETF daily adjusted closes. Tiingo (TIINGO_API_KEY) is preferred; Alpaca keys are an optional fallback."},
            "tiingo": {"configured": svc.data.tiingo.configured, "usable": svc.data.tiingo.check()[0], "status": svc.data.tiingo.check()[1]},
            "alpaca": {"configured": svc.data.alpaca.configured, "usable": svc.data.alpaca.check()[0],
                       "status": svc.data.alpaca.check()[1], "feed": svc.data.alpaca.feed.upper(),
                       "note": "Historical daily bars only; any valid ticker is accepted when the credentials work."}},
        "variables": reg.public_catalog(),
        "ticker_rules": {"pattern": R.TICKER_RE.pattern, "max_tickers": 12},
        "macro_series_allowlist": [v.id for v in R.FRED_VARIABLES],
        "transforms": {"price": list(R.KIND_TRANSFORMS["price"]), "rate": list(R.KIND_TRANSFORMS["rate"]),
                       "index": list(R.KIND_TRANSFORMS["index"]), "spread": list(R.KIND_TRANSFORMS["spread"])},
        "lookback_years": list(LOOKBACKS), "max_hypotheses": {"min": 1, "max": MAX_HYPOTHESES_CAP, "default": 3},
        "max_lag_days": 5, "example_themes": EXAMPLE_THEMES, "default_theme": DEFAULT_THEME,
        "modes": [
            {"id": "autonomous", "live": True, "requires": ["gemini"], "description": "Gemini proposes hypotheses from the catalogue."},
            {"id": "guided", "live": True, "requires": ["gemini"], "description": "Gemini translates your question into hypotheses."},
            {"id": "demo_autonomous", "live": False, "requires": [],
             "description": "Scripted hypotheses on SYNTHETIC data. Offline demo/test only."}],
        "timing_convention": "Daily observation dated t is a market close known at t; FRED publication delay is not modelled "
                             "and monthly macro series are excluded (no point-in-time vintages).",
    })


# ------------------------------------------------------------------ request validation
def parse_request(body, svc, reasoner):
    if not isinstance(body, dict):
        raise RequestError("JSON object body required")
    extra = set(body) - ALLOWED_KEYS
    if extra:
        raise RequestError(f"unsupported fields: {sorted(extra)}")
    mode = body.get("mode")
    if mode not in MODES:
        raise RequestError("mode must be one of: " + ", ".join(MODES))
    reg = svc.registry
    lookback = body.get("lookback_years", 3)
    if isinstance(lookback, bool) or lookback not in LOOKBACKS:
        raise RequestError(f"lookback_years must be one of {list(LOOKBACKS)}")
    budget = body.get("max_hypotheses", 2 if mode == "guided" else 3)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= MAX_HYPOTHESES_CAP:
        raise RequestError(f"max_hypotheses must be an integer 1..{MAX_HYPOTHESES_CAP}")
    req = {"mode": mode, "lookback_years": lookback, "max_hypotheses": budget}
    uni = body.get("universe") or {}
    if not isinstance(uni, dict) or set(uni) - {"tickers", "macro_series"}:
        raise RequestError("universe must be {tickers: [...], macro_series: [...]}")
    if mode == "demo_autonomous":
        if uni:
            raise RequestError("demo_autonomous uses fixed synthetic data and takes no universe")
        req["universe"] = {"tickers": [], "macro_series": []}
        return req
    if not reasoner.configured:
        raise RequestError("Gemini is not configured: set GEMINI_API_KEY in backend/.env to use live "
                           f"'{mode}' mode, or use mode 'demo_autonomous' (offline, synthetic).", 503)
    tickers, macros = uni.get("tickers", []), uni.get("macro_series", [])
    if not isinstance(tickers, list) or not isinstance(macros, list) or len(tickers) > 12 or len(macros) > 8:
        raise RequestError("universe.tickers (max 12) and universe.macro_series (max 8) must be lists")
    ok_t, ok_m = [], []
    for t in tickers:
        v, why = reg.resolve(t) if isinstance(t, str) else (None, "ticker must be a string")
        if v is None or why or v.source != "equity":
            raise RequestError(f"ticker {str(t)[:20]!r} rejected: {why or 'not a stock/ETF ticker'}")
        ok_t.append(v.id)
    for m in macros:
        v, why = reg.resolve(m) if isinstance(m, str) else (None, "series id must be a string")
        if v is None or why or v.source != "fred":
            raise RequestError(f"macro series {str(m)[:20]!r} rejected: {why or 'not in the FRED allowlist'}")
        ok_m.append(v.id)
    req["universe"] = {"tickers": ok_t, "macro_series": ok_m}
    if mode == "guided":
        q = body.get("question")
        if not isinstance(q, str) or not 10 <= len(q.strip()) <= 500:
            raise RequestError("guided mode requires 'question' (10-500 characters)")
        req["question"] = q.strip()
    else:
        theme = body.get("theme") or DEFAULT_THEME
        if not isinstance(theme, str) or len(theme) > 200:
            raise RequestError("theme must be a string of at most 200 characters")
        req["theme"] = theme.strip()
    return req


def _tracking(run_id):
    return f"/api/research/runs/{run_id}"


@research_bp.post("/api/research/runs")
def create_run():
    if request.content_length and request.content_length > MAX_BODY_BYTES:
        return jsonify({"error": "request body too large"}), 413
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object body required"}), 400
    st = _state()
    store = st["store"]
    # ---- legacy preset path (synthetic, schema 1.0)
    if "preset_id" in data and "mode" not in data:
        preset_id = data.get("preset_id")
        if not isinstance(preset_id, str) or preset_id not in PRESETS:
            return jsonify({"error": "preset_id must be one of: " + ", ".join(PRESETS)}), 400
        if set(data) - {"preset_id"}:
            return jsonify({"error": "only preset_id is accepted"}), 400
        run_id = str(uuid.uuid4())
        store.create(run_id, preset_id, new_report(run_id, preset_id))
        job = (store, run_id, preset_id, st["reasoner"], st["timeout_s"])
        fn = run_pipeline
    else:
        try:
            req = parse_request(data, st["services"], st["reasoner"])
        except RequestError as e:
            return jsonify({"error": e.message}), e.status
        if store.active_count() >= MAX_ACTIVE_RUNS:
            return jsonify({"error": f"too many active runs (max {MAX_ACTIVE_RUNS}); retry shortly"}), 429
        run_id = str(uuid.uuid4())
        store.create(run_id, req["mode"], new_report_v2(run_id, req["mode"], req))
        job = (store, run_id, st["services"], st["timeout_s"])
        fn = run_autonomous
    if st["sync"]:
        fn(*job)
    else:
        st["pool"].submit(fn, *job)
    return jsonify({"run_id": run_id, "status": "queued", "tracking_url": _tracking(run_id)}), 202


@research_bp.get("/api/research/runs/<run_id>")
def get_run(run_id):
    rep = _state()["store"].get(run_id)
    if rep is None:
        return jsonify({"error": "run not found"}), 404
    return jsonify(rep)


@research_bp.get("/api/research/runs")
def list_runs():
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 200))
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400
    rows = _state()["store"].list(limit, offset)
    for r in rows:
        r["mode"] = r["preset_id"] if r["preset_id"] in MODES else "demo"
    return jsonify({"runs": rows, "limit": limit, "offset": offset})


@research_bp.post("/api/research/runs/<run_id>/simulate")
def simulate(run_id):
    """Paper-trade the frozen strategy of one hypothesis and compare it with holding the S&P 500.
    Replays stored research data; places no orders."""
    st = _state()
    rep = st["store"].get(run_id)
    if rep is None:
        return jsonify({"error": "run not found"}), 404
    if not str(rep.get("schema_version", "")).startswith("2"):
        return jsonify({"error": "simulation is only available for autonomous / guided / demo_autonomous runs"}), 400
    if rep["status"] != "completed":
        return jsonify({"error": f"run is '{rep['status']}'; simulate after it completes"}), 409
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) - {"hypothesis_id", "capital", "period", "cost_bps"}:
        return jsonify({"error": "body must be {hypothesis_id, capital?, period?, cost_bps?}"}), 400
    hid = body.get("hypothesis_id")
    cap, period, cost = body.get("capital", 10_000), body.get("period", "holdout"), body.get("cost_bps", 2.0)
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")
    if not isinstance(hid, str):
        return jsonify({"error": "hypothesis_id is required"}), 400
    if not num(cap) or not MIN_CAPITAL <= cap <= MAX_CAPITAL:
        return jsonify({"error": f"capital must be a number between {MIN_CAPITAL:,.0f} and {MAX_CAPITAL:,.0f}"}), 400
    if period not in PERIODS:
        return jsonify({"error": "period must be one of: " + ", ".join(PERIODS)}), 400
    if not num(cost) or not 0 <= cost <= 50:
        return jsonify({"error": "cost_bps must be a number between 0 and 50"}), 400
    svc = st["services"]
    reg = svc.registry
    lookback = rep["request"].get("lookback_years", 3)
    want = benchmark_variable_id(rep["mode"], svc.data.equity_status()[0])
    levels, err = None, None
    for vid in dict.fromkeys([want, "SP500"] if rep["mode"] != "demo_autonomous" else [want]):
        v, why = reg.resolve(vid)
        if v is None or why:
            continue
        try:
            levels = (svc.data.get(v, lookback)[0], v)
            break
        except ProviderError as e:
            err = str(e)
    if levels is None:
        return jsonify({"error": f"Benchmark data unavailable: {err or 'no provider'}"}), 502
    series, v = levels
    note = ("SYNTHETIC market index standing in for the S&P 500 (demo mode)." if v.source == "fixture"
            else "S&P 500 price index from FRED: price return only, dividends excluded." if v.id == "SP500"
            else "SPDR S&P 500 ETF (SPY), split/dividend-adjusted.")
    label = {"variable_id": v.id, "label": "Synthetic market index" if v.source == "fixture" else "S&P 500 (buy and hold)", "note": note}
    try:
        out = run_simulation(rep, hid, series, label, capital=float(cap), period=period, cost_bps=float(cost))
    except SimulationError as e:
        return jsonify({"error": e.message}), e.status
    return jsonify(to_jsonable(out))


from . import globe_api  # noqa: E402,F401  (registers /api/research/globe/* on research_bp)

from . import pairs_api  # noqa: E402,F401  (registers /api/research/pairs*)
