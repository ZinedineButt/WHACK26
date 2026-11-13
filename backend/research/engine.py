"""Autonomous / guided / demo research run: generation -> screening ->
pre-registration -> (correlation -> causality -> validation) per hypothesis ->
family-wise correction -> narrative. Saved after every step."""
import copy
import datetime as dt
import hashlib
import json
import re
import time
import traceback

from . import autollm, backtest as BT, criteria as C, hstages as H, literature as LIT, peers as PEERS, registry as R, validation as V
from . import strength as ST
from .demo_literature import DEMO_LITERATURE, demo_lit
from .hypothesis import HypothesisProposal, validate_proposal
from .llm import SOURCE_FALLBACK, SOURCE_GEMINI, SOURCE_TEMPLATE
from .providers import ProviderError
from .store import to_jsonable

SCHEMA_VERSION = "2.0"
STAGE_DEFS = [("correlation", "Measuring association"), ("causality", "Investigating causal explanations"),
              ("validation", "Adversarial validation"), ("backtest", "Backtesting the frozen rule"),
              ("generalisation", "Re-testing on peer assets"), ("critique", "Literature review and self-critique")]
MARKET_CONTROL_CANDIDATES = {"live": ["SPY", "SP500"], "demo": ["SYN_MKT"]}
MAX_HYPOTHESES_CAP = 5
NO_CORRELATION_LABEL = "Initial hypotheses with no correlation"
RATES_TOPIC = re.compile(r"yield|rate\b|rates|treasur|bond|credit|\bfed\b|interest|inflation|spread|monetary|duration")
MAX_CANDIDATES = 8


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def new_report(run_id, mode, request):
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "id": run_id, "status": "queued", "mode": mode,
        "request": request, "current_phase": "queued", "current_hypothesis": None,
        "phases": [{"id": "hypothesis_generation", "status": "pending", "message": "Generating hypotheses"}],
        "generation": None, "preregistration": None, "hypotheses": [],
        "multiple_testing": {"proposed": 0, "attempted": 0, "correction": "holm", "family_alpha": C.ALPHA},
        "visualization_data": {}, "summary": {"overall": "", "hypotheses": []},
        "reasoning_source": None, "reasoning_sources": {}, "data_provenance": "synthetic" if mode == "demo_autonomous" else "real",
        "warnings": [], "errors": [],
        "timestamps": {"created": now(), "started": None, "finished": None, "updated": now()},
    }


def _hyp_shell(hid, raw):
    return {"hypothesis_id": hid, "status": "pending", "proposal": raw, "normalized": None,
            "eligibility": {"accepted": False, "reasons": [], "notes": []},
            "stages": [{"id": i, "status": "pending", "message": m} for i, m in STAGE_DEFS],
            "provenance": None, "correlation": None, "causality": None, "validation": None,
            "backtest": None, "generalisation": None, "literature": None, "literature_prior": None, "critique": None,
            "verdict": None, "warnings": [], "error": None}


def _stage(h, sid, status, msg=None):
    for s in h["stages"]:
        if s["id"] == sid:
            s["status"] = status
            if msg:
                s["message"] = msg


class Run:
    def __init__(self, store, run_id, svc, timeout_s):
        self.store, self.svc, self.timeout_s = store, svc, timeout_s
        self.rep = store.get(run_id)
        self.t0 = time.time()

    def save(self, phase=None, hyp="__keep__"):
        if phase:
            self.rep["current_phase"] = phase
        if hyp != "__keep__":
            self.rep["current_hypothesis"] = hyp
        self.rep["timestamps"]["updated"] = now()
        self.store.save(self.rep)

    def check(self):
        if time.time() - self.t0 > self.timeout_s:
            raise TimeoutError(f"run exceeded {self.timeout_s}s")


def _reasoner(run):
    """Demo mode is deterministic/offline: never calls Gemini."""
    return run.svc.offline_reasoner if run.rep["mode"] == "demo_autonomous" else run.svc.reasoner


def _fail(run, code, message, stage=None):
    run.rep["status"] = "failed"
    run.rep["errors"].append({"code": code, "message": message, "phase": stage or run.rep["current_phase"]})
    run.rep["timestamps"]["finished"] = now()
    run.save()


def run_autonomous(store, run_id, svc, timeout_s=600):
    run = Run(store, run_id, svc, timeout_s)
    rep = run.rep
    try:
        rep["status"] = "running"
        rep["timestamps"]["started"] = now()
        _generate(run)
        if rep["status"] == "failed":
            return
        _screen_and_freeze(run)
        if rep["status"] == "failed":
            return
        frames = run.frames
        for h in rep["hypotheses"]:
            if h["status"] == "pending":
                _run_hypothesis(run, h, frames[h["hypothesis_id"]])
        _family_correction(run)
        _finish(run)
    except Exception as e:
        traceback.print_exc()
        rep = store.get(run_id) or rep
        run.rep = rep
        _fail(run, "timeout" if isinstance(e, TimeoutError) else "internal_error", f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ phase 0
def _generate(run):
    rep, svc, req = run.rep, run.svc, run.rep["request"]
    mode = rep["mode"]
    reg = svc.registry
    rep["phases"][0]["status"] = "running"
    run.save("hypothesis_generation", None)
    budget = max(1, min(int(req.get("max_hypotheses", 3)), MAX_HYPOTHESES_CAP))
    fixtures = mode == "demo_autonomous"
    ncand = budget if fixtures else min(budget * 2, MAX_CANDIDATES)     # over-generate, then keep the most viable
    background = None
    uni = req.get("universe") or {}
    allowed = None
    if fixtures:
        variables = reg.fixture_catalog()
        raws, unsupported, source = autollm.DEMO_HYPOTHESES[:budget], "", SOURCE_TEMPLATE
    else:
        names = [t.upper() for t in uni.get("tickers", [])] + [m.upper() for m in uni.get("macro_series", [])]
        if names:
            allowed = set(names)
            variables = [v for v in (reg.resolve(n)[0] for n in names) if v is not None]
        else:
            variables = reg.live_catalog()
        note = (f"Restrict the hypotheses to these variables: {sorted(allowed)}.\n" if allowed else "")
        topic = req.get("question") or req.get("theme") or ""
        background = LIT.search(None, limit=6, query=f"{topic} stock market relationship")
        if background["status"] == "unavailable":
            rep["warnings"].append("Background literature search failed; hypotheses were generated without it.")
        try:
            raws, unsupported = autollm.propose(svc.reasoner, mode, variables, ncand, req.get("lookback_years", 3),
                                                req.get("theme", ""), req.get("question", ""), note, background["papers"],
                                                tickers_ok=reg.equity_available)
            source = SOURCE_GEMINI
        except Exception as e:
            rep["phases"][0]["status"] = "failed"
            _fail(run, "gemini_error", f"Hypothesis generation failed ({type(e).__name__}: {str(e)[:200]}). "
                  "No fallback is used for live modes, so no research was run.", "hypothesis_generation")
            return
    rep["generation"] = {"source": source, "model": svc.reasoner.model if source == SOURCE_GEMINI else None,
                         "requested_budget": budget, "candidate_pool": ncand, "proposals_received": len(raws),
                         "unsupported_reason": unsupported, "catalog": [v.id for v in variables],
                         "background_literature": background}
    rep["reasoning_source"] = source
    rep["reasoning_sources"] = {"generation": source, "causal": [], "narrative": []}
    accepted = 0
    for i, raw in enumerate(raws, 1):
        hid = f"H{i}"
        try:
            prop = HypothesisProposal.model_validate({**raw, "hypothesis_id": hid})
            norm, reasons, notes = validate_proposal(prop, reg, allowed, fixtures)
        except Exception as e:
            norm, reasons, notes = None, [f"schema violation: {type(e).__name__}"], []
        h = _hyp_shell(hid, raw)
        h["eligibility"]["notes"] = notes
        if norm is None:
            h["status"], h["eligibility"]["reasons"] = "rejected", reasons
        elif accepted >= ncand:
            h["status"], h["eligibility"]["reasons"] = "rejected", [f"over the candidate limit ({ncand})"]
        else:
            h["normalized"], h["eligibility"]["accepted"] = norm, True
            accepted += 1
        rep["hypotheses"].append(h)
    rep["multiple_testing"]["proposed"] = len(raws)
    rep["phases"][0].update(status="completed", message=f"{len(raws)} proposed, {accepted} passed validation")
    rep["multiple_testing"]["candidates_after_validation"] = accepted
    run.save()
    if accepted == 0:
        why = unsupported or "; ".join(r for h in rep["hypotheses"] for r in h["eligibility"]["reasons"][:2]) or "no proposals"
        hint = ""
        if not reg.equity_available and not fixtures:
            hint = (f" Note: stock/ETF data is unavailable right now ({reg.equity_reason}). Questions about companies, sectors or themes "
                    "need a stock data provider (set TIINGO_API_KEY in backend/.env.local); only macro series and the S&P 500 / NASDAQ / DJIA indices can be tested without them.")
        _fail(run, "no_valid_hypotheses", f"No testable hypotheses: {why}{hint}", "hypothesis_generation")


# ------------------------------------------------------------------ screening + freeze
def _market_control(run, h, levels):
    cands = MARKET_CONTROL_CANDIDATES["demo" if run.rep["mode"] == "demo_autonomous" else "live"]
    own = {h["normalized"]["predictor"]["variable_id"], h["normalized"]["outcome"]["variable_id"]}
    for c in cands:
        if c in own:
            continue
        v, why = run.svc.registry.resolve(c)
        if v is None or why:
            continue
        try:
            if c not in levels:
                levels[c] = run.svc.data.get(v, run.rep["request"].get("lookback_years", 3))
            return c
        except ProviderError:
            continue
    return None


def _screen_and_freeze(run):
    rep, svc = run.rep, run.svc
    reg, lookback = svc.registry, rep["request"].get("lookback_years", 3)
    budget = max(1, min(int(rep["request"].get("max_hypotheses", 3)), MAX_HYPOTHESES_CAP))
    live = rep["mode"] != "demo_autonomous"
    levels, run.frames = {}, {}
    run.levels = levels
    # 1) data availability FIRST: candidates whose data cannot be fetched never consume the budget
    for h in rep["hypotheses"]:
        if not h["eligibility"]["accepted"]:
            continue
        run.check()
        n = h["normalized"]
        try:
            for vid in n["required_datasets"]:
                if vid not in levels:
                    levels[vid] = svc.data.get(reg.resolve(vid)[0], lookback)
            mc = _market_control(run, h, levels)
            h["normalized"]["market_control"] = [mc] if mc and mc not in n["control_variables"] else []
            controls = {}
            for c in n["control_variables"] + h["normalized"]["market_control"]:
                controls[c] = reg.resolve(c)[0].default_transform
            series = {k: v[0] for k, v in levels.items()}
            df, info = H.build_frame(n, series, controls)
        except (ProviderError, ValueError) as e:
            h["status"] = "rejected"
            h["eligibility"].update(accepted=False, reasons=[f"data screening failed: {e}"])
            continue
        h["eligibility"]["data_checked"] = True
        _, b_ = V.split_indices(len(df))
        h["provenance"] = {
            "data_type": "synthetic" if not live else "real",
            "variables": [levels[v][1] for v in sorted({*n["required_datasets"], *controls})],
            "transformations": {"predictor": n["predictor"]["transform"], "outcome": n["outcome"]["transform"],
                                "controls": controls},
            "timing": "predictor observed at the close of day t; outcome measured strictly after t",
            "alignment": "levels inner-joined on common observation dates BEFORE transforming; no look-ahead",
            "sample_start": str(df.index[0].date()), "sample_end": str(df.index[-1].date()), "observations": int(len(df)),
            "development_n": int(b_), "holdout_n": int(len(df) - b_), **info}
        run.frames[h["hypothesis_id"]] = df
    viable = [h for h in rep["hypotheses"] if h["eligibility"]["accepted"]]
    rep["multiple_testing"]["candidates_with_data"] = len(viable)
    if not viable:
        _fail(run, "no_testable_hypotheses", "All proposals failed validation or data screening: " + "; ".join(
            f"{h['hypothesis_id']}: {h['eligibility']['reasons'][0]}" for h in rep["hypotheses"] if h["eligibility"]["reasons"]),
            "hypothesis_generation")
        return

    # 2) literature viability: keep the hypotheses that research suggests are realistic
    if live:
        run.save("literature_review", None)
        cands = []
        for i, h in enumerate(viable, 1):
            run.check()
            lit = LIT.search(h["normalized"])
            h["literature"] = lit
            cands.append({"index": i, "question": h["normalized"]["question"], "predictor": h["normalized"]["predictor"]["label"],
                          "outcome": h["normalized"]["outcome"]["label"], "mechanism": h["normalized"]["mechanism"], "papers": lit["papers"]})
        ranks, src, warns = autollm.rank_candidates(svc.reasoner, cands, topic=(rep["request"].get("question") or rep["request"].get("theme") or ""))
        rep["reasoning_sources"]["ranking"] = src
        rep["warnings"] += warns
        for i, h in enumerate(viable, 1):
            h["literature_prior"] = {**ranks[i], "papers_retrieved": len(h["literature"]["papers"]), "search_status": h["literature"]["status"]}
        order = sorted(enumerate(viable), key=lambda t: (-t[1]["literature_prior"]["viability"], t[0]))
        topic = (rep["request"].get("question") or rep["request"].get("theme") or "").lower()
        rates_topic = bool(RATES_TOPIC.search(topic))

        def family(h):
            v = reg.resolve(h["normalized"]["predictor"]["variable_id"])[0]
            return "rates_credit" if (v and v.category in ("rates", "credit") and not rates_topic) else h["normalized"]["predictor"]["variable_id"]
        keep_ids, used = [], set()
        pool = []
        for pos, h in order:                                   # unrealistic ones never compete
            lp = h["literature_prior"]
            if lp["source"] == SOURCE_GEMINI and lp["viability"] <= 1:
                h["status"] = "rejected"
                h["eligibility"].update(accepted=False, reasons=["judged unrealistic from the related research (viability 1/5): " + lp["summary"][:200]])
            else:
                pool.append(h)
        for h in pool:                                         # pass 1: best viability, one hypothesis per predictor family
            if len(keep_ids) < budget and family(h) not in used:
                keep_ids.append(h["hypothesis_id"]); used.add(family(h))
        for h in pool:                                         # pass 2: fill any remaining slots by viability
            if len(keep_ids) < budget and h["hypothesis_id"] not in keep_ids:
                keep_ids.append(h["hypothesis_id"])
        for h in pool:
            if h["hypothesis_id"] not in keep_ids:
                same = any(family(x) == family(h) for x in pool if x["hypothesis_id"] in keep_ids)
                h["status"] = "rejected"
                h["eligibility"].update(accepted=False, reasons=[
                    f"viable, but ranked below the {budget} most realistic hypotheses by literature viability "
                    f"({h['literature_prior']['viability']}/5)" + ("; diversity rule: a hypothesis on the same predictor family was already selected" if same else "")])
        selected = [h for h in viable if h["hypothesis_id"] in keep_ids]
    else:
        selected = viable[:budget]
        for h in viable:
            d = DEMO_LITERATURE.get(h["normalized"]["predictor"]["variable_id"])
            if d:
                h["literature_prior"] = dict(d["prior"])
        for h in viable[budget:]:
            h["status"] = "rejected"
            h["eligibility"].update(accepted=False, reasons=[f"over the hypothesis budget ({budget})"])
    for h in rep["hypotheses"]:
        if h["status"] == "rejected":
            run.frames.pop(h["hypothesis_id"], None)
    if not selected:
        _fail(run, "no_testable_hypotheses", "No hypothesis survived literature viability screening.", "hypothesis_generation")
        return

    # 3) freeze the plan of the selected hypotheses BEFORE any statistic
    planned = []
    for h in selected:
        n, df = h["normalized"], run.frames[h["hypothesis_id"]]
        _, b_ = V.split_indices(len(df))                      # b_ = end of the development sample (75%)
        n["peer_candidates"] = PEERS.candidates(n["outcome"]["variable_id"], n["predictor"]["variable_id"], reg) \
            if reg.resolve(n["outcome"]["variable_id"])[0].kind == "price" else []
        planned.append({"hypothesis_id": h["hypothesis_id"], "plan": {
            "predictor": n["predictor"], "outcome": n["outcome"], "primary_lag": n["lag_days"], "direction": n["direction"],
            "controls": list(h["provenance"]["transformations"]["controls"]), "lag_family": [0] if H.is_forward(n) else H.FAMILY_LAGS,
            "tests": "lagged_correlation_hac -> (stop if no correlation) -> causal checks -> validation -> backtest -> peers -> critique",
            "backtest": {"rule": "long/short sign(predictor) at t, earn outcome at t+lag", "cost_bps": C.COST_BPS,
                         "sign": "pre-registered direction (development-sample sign if 'any')"},
            "peer_candidates": n["peer_candidates"],
            "peer_rule": "peers (fixed server-side map) are tested only if the primary association is detected",
            "stop_rule": "hypotheses with no detected correlation stop after stage 1",
            "viability_score": (h["literature_prior"] or {}).get("viability"),
            "split": {"development": [str(df.index[0].date()), str(df.index[b_ - 1].date())],
                      "holdout": [str(df.index[b_].date()), str(df.index[-1].date())]}}})
    body = json.dumps(to_jsonable(planned), sort_keys=True)
    rep["preregistration"] = {
        "frozen_at": now(), "sha256": hashlib.sha256(body.encode()).hexdigest(), "hypotheses": to_jsonable(planned),
        "family_size": len(planned), "correction": "holm", "alpha": C.ALPHA,
        "selection": ("Candidates were over-generated, screened for data availability, then ranked by literature viability; "
                      f"the top {budget} were frozen." if live else "Scripted demo hypotheses."),
        "holdout": "last 25% of each sample; evaluated exactly once, after all hypotheses and methods were frozen",
        "label": "Stages 1-2 are exploratory (development sample). Only the holdout is confirmatory.",
        "frozen_before_any_statistic": True}
    rep["multiple_testing"]["attempted"] = len(planned)
    rep["warnings"].append("Exploratory findings: hypotheses were generated and screened on the same data universe.")
    run.save("preregistered")


# ------------------------------------------------------------------ per hypothesis
def _run_hypothesis(run, h, df):
    rep, svc = run.rep, run.svc
    n = h["normalized"]
    hid = h["hypothesis_id"]
    current = None
    try:
        h["status"] = "running"
        dev, _ = H.split_frame(df)
        current = "correlation"
        _stage(h, current, "running"); run.save("correlation", hid)
        h["correlation"] = H.stage1(dev, n)
        _stage(h, current, "completed", "Computed association statistics"); run.save(); run.check()

        c = h["correlation"]
        if c["association_verdict"] == "not_detected":
            # No correlation: stop here. Nothing downstream is meaningful for a null.
            rep["visualization_data"].setdefault(hid, {}).update(H.viz(df, n, c, None))
            for sid in ("causality", "validation", "backtest", "generalisation", "critique"):
                _stage(h, sid, "skipped", "Stopped: no correlation")
            h["status"] = "no_correlation"
            h["verdict"] = {"association": "not_detected", "causal": "not_run", "reliability": "not_run", "backtest": "not_run",
                            "generalisation": "not_run", "confidence": "low", "evidence_level": "exploratory",
                            "holdout_confirmation": "not_run",
                            "statement": "Initial hypothesis with no correlation: testing stopped before the causal and validation stages."}
            run.save()
            return

        current = "causality"
        _stage(h, current, "running"); run.save("causality", hid)
        c = h["correlation"]
        facts = {"r": c["r"], "n": c["n"], "p_hac": c["p_value_hac"], "association_verdict": c["association_verdict"]}
        nodes = {n["predictor"]["variable_id"], n["outcome"]["variable_id"], *n["control_variables"], *n.get("market_control", [])}
        proposal, src, warns = autollm.causal_proposal(_reasoner(run), {**n}, facts, nodes)
        rep["reasoning_sources"]["causal"].append({"hypothesis_id": hid, "source": src})
        h["warnings"] += warns
        h["causality"] = H.stage2(dev, n, proposal, c, extras=_causal_extras(run, n, reg_live=rep["mode"] != "demo_autonomous"))
        rep["visualization_data"].setdefault(hid, {})["causal_graph"] = H.causal_graph(n, h["causality"], proposal)
        _stage(h, current, "completed", "Ran causal checks"); run.save(); run.check()

        current = "validation"
        _stage(h, current, "running"); run.save("validation", hid)
        h["validation"] = H.stage3(df, n)
        vd = H.viz(df, n, c, h["validation"])
        rep["visualization_data"][hid].update(vd)
        _stage(h, current, "completed", "Finished validation battery")
        run.save(); run.check()

        current = "backtest"
        _stage(h, current, "running"); run.save("backtest", hid)
        bt = BT.run_backtest(df, n, dev_r=c["r"])
        rep["visualization_data"][hid]["backtest_equity"] = bt.pop("equity_curve", [])
        h["backtest"] = bt
        _stage(h, current, "completed" if bt["applicable"] else "skipped",
               "Backtested the frozen rule" if bt["applicable"] else "Not applicable: " + bt["reason"])
        run.save(); run.check()

        current = "generalisation"
        _stage(h, current, "running"); run.save("generalisation", hid)
        sign = BT.frozen_sign(n, c["r"])
        if c["association_verdict"] != "detected":
            h["generalisation"] = {"status": "not_run", "verdict": "not_run", "peers": [], "skipped": [],
                                   "verdict_rule": "Peers are only tested when the primary association is detected (pre-registered rule)."}
            _stage(h, current, "skipped", "Skipped: primary association not detected")
        elif not n.get("peer_candidates"):
            h["generalisation"] = {"status": "not_run", "verdict": "not_run", "peers": [], "skipped": [],
                                   "verdict_rule": "No peer assets are defined for this outcome."}
            _stage(h, current, "skipped", "Skipped: no peers defined for this outcome")
        else:
            h["generalisation"] = PEERS.run_peers(n, n["peer_candidates"], svc, rep["request"].get("lookback_years", 3), run.levels, sign)
            _stage(h, current, "completed" if h["generalisation"]["status"] == "completed" else "skipped",
                   f"Peers: {h['generalisation']['verdict']}")
        rep["visualization_data"][hid]["peers"] = h["generalisation"]["peers"]
        h["status"] = "completed"
        run.save(); run.check()
    except TimeoutError:
        raise
    except Exception as e:
        traceback.print_exc()
        if current:
            _stage(h, current, "failed")
        h["status"] = "failed"
        h["error"] = {"code": "internal_error", "stage": current, "message": f"{type(e).__name__}: {e}"}
        rep["errors"].append({"code": "hypothesis_failed", "hypothesis_id": hid, "message": h["error"]["message"]})
        run.save()


# ------------------------------------------------------------------ family correction + finish
def _family_correction(run):
    rep = run.rep
    done = [h for h in rep["hypotheses"] if h["correlation"] is not None]
    if not done:
        return
    from . import stats as S
    ps = [h["correlation"]["p_value_hac"] for h in done]
    adj, rej = S.holm(ps, C.ALPHA)
    mt = rep["multiple_testing"]
    mt.update(correction="holm", primary_p_hac={h["hypothesis_id"]: p for h, p in zip(done, ps)},
              primary_p_holm={h["hypothesis_id"]: a for h, a in zip(done, adj)},
              significant_after_correction=[h["hypothesis_id"] for h, r in zip(done, rej) if r],
              hypotheses_in_family=len(done),
              within_hypothesis_lag_family=H.FAMILY_LAGS,
              note=("Holm correction over the primary-lag HAC p-value of every hypothesis actually tested in this run "
                    "(development sample). Each hypothesis additionally Holm-corrects over its pre-registered lag family. "
                    "The same p-values are not counted as independent evidence."))
    for h, a, r in zip(done, adj, rej):
        v = h["validation"]
        if not v or not v.get("tests"):
            continue
        v["tests"].append({
            "id": "family_multiple_testing", "name": "Run-level Holm correction across all hypotheses tested",
            "metrics": {"p_hac": h["correlation"]["p_value_hac"], "p_holm": a, "family_size": len(done)},
            "threshold": f"Holm p<{C.ALPHA}", "result": "pass" if r else "fail",
            "explanation": ("Survives correction for the whole hypothesis family." if r else
                            "Does not survive correction for the number of hypotheses tested."),
            "limitations": ["Family = hypotheses generated in this run; earlier runs are not counted."], "critical": True})
        v["reliability_verdict"], v["verdict_rule"] = V.verdict_from_tests(v["tests"])
        v["strength"] = ST.reliability_strength(v["tests"])
    run.save("family_correction", None)


def _facts(h):
    c, ca, v = h["correlation"], h["causality"], h["validation"]
    return to_jsonable({
        "correlation": {k: c[k] for k in ("x_label", "y_label", "r", "n", "association_verdict")},
        "causality": {k: ca[k] for k in ("causal_verdict", "verdict_rule")},
        "validation": {"reliability_verdict": v["reliability_verdict"], "verdict_rule": v["verdict_rule"],
                       "tests": [{"id": t["id"], "result": t["result"]} for t in v["tests"]]}})


def _critique_facts(h):
    c, ca, v = h["correlation"], h["causality"], h["validation"]
    bt, g = h["backtest"] or {}, h["generalisation"] or {}
    hold = bt.get("holdout") or {}
    return to_jsonable({
        "hypothesis": h["normalized"]["question"],
        "association_verdict": c["association_verdict"], "causal_verdict": ca["causal_verdict"],
        "reliability_verdict": v["reliability_verdict"], "backtest_verdict": bt.get("verdict", "not_run"),
        "generalisation_verdict": g.get("verdict", "not_run"),
        "r": c["r"], "n": c["n"], "p_hac": c["p_value_hac"],
        "failed_tests": [t["id"] for t in v["tests"] if t["result"] == "fail"],
        "passed_tests": [t["id"] for t in v["tests"] if t["result"] == "pass"],
        "not_run_checks": [t["id"] for t in ca["tests"] if t["result"] == "not_run"],
        "all_tests": [t["id"] for t in v["tests"] + ca["tests"]],
        "backtest_holdout": {k: hold.get(k) for k in ("sharpe", "hit_rate", "total_return_pct", "mean_daily_net_pct", "n")},
        "peers": {"n": g.get("n_peers"), "share_same_sign": g.get("share_same_sign"),
                  "share_significant": g.get("share_significant_same_sign")}})


def _causal_extras(run, n, reg_live):
    """Inputs for the negative-control and event checks. Live runs only; failures degrade to 'not_run' with a reason."""
    if not reg_live:
        return {"live": False}
    out = {"live": True, "neg_series": None}
    svc, reg = run.svc, run.svc.registry
    lookback = run.rep["request"].get("lookback_years", 3)
    used = {n["predictor"]["variable_id"], n["outcome"]["variable_id"], *n["control_variables"], *n.get("market_control", [])}
    for vid in H.NEG_CONTROL_CANDIDATES:
        if vid in used:
            continue
        try:
            var = reg.resolve(vid)[0]
            level = svc.data.get(var, lookback)[0]
            series = H.T.apply(level.dropna(), n["outcome"]["transform"])
            series = series.replace([float("inf"), float("-inf")], float("nan")).dropna()
            out.update(neg_series=series, neg_id=vid, neg_label=var.label)
            return out
        except Exception as e:                                      # noqa: BLE001  never let a side check break the run
            out["neg_note"] = f"Negative-control series {vid} could not be loaded ({type(e).__name__})."
    return out


def _critique_phase(run):
    rep = run.rep
    live = rep["mode"] != "demo_autonomous"
    run.save("critique", None)
    for h in rep["hypotheses"]:
        if h["status"] != "completed":
            continue
        run.check()
        hid = h["hypothesis_id"]
        run.rep["current_hypothesis"] = hid
        _stage(h, "critique", "running"); run.save()
        if live:
            lit = h["literature"] if (h.get("literature") and h["literature"]["status"] == "completed") else LIT.search(h["normalized"])
        else:
            lit = demo_lit(h["normalized"]["predictor"]["variable_id"])
        h["literature"] = lit
        facts = _critique_facts(h)
        crit, src, warns = autollm.self_critique(_reasoner(run), facts, lit["papers"])
        rep["reasoning_sources"].setdefault("critique", []).append({"hypothesis_id": hid, "source": src})
        h["warnings"] += warns
        h["critique"] = crit
        if lit["status"] == "unavailable":
            h["warnings"].append(lit["note"])
        _stage(h, "critique", "completed", f"{len(lit['papers'])} papers reviewed; confidence {crit['confidence']}")
        run.save()


def _plain_overall(rep, counts, stopped, tested):
    mt, budget = rep["multiple_testing"], rep["request"].get("max_hypotheses")
    parts = []
    if mt["proposed"] > mt["attempted"]:
        parts.append(f"The AI proposed {mt['proposed']} candidate ideas"
                     + (f" and you asked for at most {budget} to be tested" if budget else "")
                     + f". Only the {mt['attempted']} best supported by related research were tested; the rest were set aside, not run.")
    else:
        parts.append(f"{mt['attempted']} {'idea was' if mt['attempted'] == 1 else 'ideas were'} tested.")
    if stopped:
        parts.append(f"{len(stopped)} showed no link in the data and stopped early ({NO_CORRELATION_LABEL.lower()}).")
    if tested:
        parts.append(f"Of the {tested} that went through every check: {counts['supported']} held up on unseen data, "
                     f"{counts['rejected']} did not, {counts['inconclusive']} unclear"
                     + (f", {counts['failed']} failed to run" if counts["failed"] else "") + ".")
    parts.append("Every result is shown, including the ones that found nothing. Holding up on unseen data is not proof of cause and effect; treat all findings as exploratory.")
    return " ".join(parts)


def _finish(run):
    rep, svc = run.rep, run.svc
    _critique_phase(run)
    lines, counts = [], {"supported": 0, "rejected": 0, "inconclusive": 0, "failed": 0, "no_correlation": 0}
    for h in rep["hypotheses"]:
        if h["status"] == "no_correlation":
            c = h["correlation"]
            counts["no_correlation"] += 1
            h["summary"] = (f"r={c['r']:.3f} at lag {c['primary_lag']} (n={c['n']}, HAC p={c['p_value_hac']:.3g}). "
                            "Initial hypothesis with no correlation: testing stopped.")
            lines.append({"hypothesis_id": h["hypothesis_id"], "question": h["normalized"]["question"], "text": h["summary"]})
            continue
        if h["status"] == "rejected":
            lines.append({"hypothesis_id": h["hypothesis_id"], "text": "Not tested: " + "; ".join(h["eligibility"]["reasons"])})
            continue
        if h["status"] != "completed":
            counts["failed"] += 1
            lines.append({"hypothesis_id": h["hypothesis_id"], "text": f"Failed during {h['error']['stage'] if h['error'] else '?'}"})
            continue
        run.check()
        nar, src, warns = _reasoner(run).narrative(_facts(h))
        rep["reasoning_sources"]["narrative"].append({"hypothesis_id": h["hypothesis_id"], "source": src})
        h["warnings"] += warns
        h["causality"]["reasoning"] = nar["causality_reasoning"]
        h["validation"]["reasoning"] = nar["validation_reasoning"]
        c, ca, v = h["correlation"], h["causality"], h["validation"]
        bt, g = h["backtest"], h["generalisation"]
        hold = next((t for t in v["tests"] if t["id"] == "holdout_single_evaluation"), None)
        h["strength"] = ST.overall(h)
        h["verdict"] = {"strength": h["strength"], "association": c["association_verdict"], "causal": ca["causal_verdict"],
                        "reliability": v["reliability_verdict"], "backtest": bt["verdict"], "generalisation": g["verdict"],
                        "confidence": h["critique"]["confidence"], "evidence_level": "exploratory",
                        "holdout_confirmation": hold["result"] if hold else "not_run",
                        "statement": (f"Association {c['association_verdict']}; causation {ca['causal_verdict'].replace('_', ' ')}; "
                                      f"reliability {v['reliability_verdict']}; backtest {bt['verdict']}; peers {g['verdict']}. "
                                      "Association alone never establishes causation.")}
        counts[v["reliability_verdict"]] += 1
        bt_txt = ""
        if bt.get("applicable") and bt["holdout"].get("sharpe") is not None:
            bt_txt = f" Holdout backtest: net Sharpe {bt['holdout']['sharpe']:.2f}, hit rate {bt['holdout']['hit_rate']:.1%}."
        lines.append({"hypothesis_id": h["hypothesis_id"], "question": h["normalized"]["question"],
                      "text": (f"r={c['r']:.3f} at lag {c['primary_lag']} (n={c['n']}, HAC p={c['p_value_hac']:.3g}); "
                               + h["verdict"]["statement"] + bt_txt)})
        h["summary"] = lines[-1]["text"]
    tested = sum(1 for h in rep["hypotheses"] if h["status"] == "completed")
    stopped = [h["hypothesis_id"] for h in rep["hypotheses"] if h["status"] == "no_correlation"]
    rep["summary"] = {
        "hypotheses": lines, "no_correlation_label": NO_CORRELATION_LABEL, "no_correlation": stopped,
        "overall": _plain_overall(rep, counts, stopped, tested),
        "counts": counts}
    srcs = {rep["reasoning_sources"]["generation"], *[x["source"] for k in ("narrative", "causal", "critique")
                                                      for x in rep["reasoning_sources"].get(k, [])]}
    if SOURCE_FALLBACK in srcs:
        rep["warnings"].append("Some Gemini steps fell back to labelled templates; see reasoning_sources.")
    if rep["mode"] == "demo_autonomous":
        rep["warnings"].append("DEMO MODE: scripted hypotheses on SYNTHETIC data. Not live research.")
    rep["status"] = "completed"
    rep["current_phase"], rep["current_hypothesis"] = "complete", None
    rep["timestamps"]["finished"] = now()
    run.save()