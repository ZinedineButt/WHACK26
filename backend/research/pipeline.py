"""Orchestrates the three stages, saving the report after each one."""
import time
import traceback

from . import criteria as C, datasets, correlation, causal, validation, stats as S
from .presets import PRESETS
from .store import to_jsonable

SCHEMA_VERSION = "1.0"
STAGES = [("correlation", "Calculating Pearson correlation"),
          ("causality", "Reviewing confounders and lead/lag tests"),
          ("validation", "Running sceptic robustness checks")]


def new_report(run_id, preset_id):
    p = PRESETS[preset_id]
    return {
        "schema_version": SCHEMA_VERSION, "id": run_id, "status": "queued", "current_stage": "queued",
        "stages": [{"id": i, "status": "pending", "message": m} for i, m in STAGES],
        "input": {"preset_id": preset_id, "label": p["label"]},
        "provenance": {"data_type": "synthetic", "reasoning_source": None, "seed": C.SEED,
                       "observations": None, "sample_start": None, "sample_end": None,
                       "frequency": "business-daily", "transformation": "synthetic simple returns (%), no price levels used",
                       "missing_data": None, "factor_label": "Synthetic market factor returns (%)"},
        "correlation": None, "causality": None, "validation": None,
        "visualization_data": {"aligned_series": [], "scatter": [], "rolling_correlation": [],
                               "lag_profile": [], "causal_graph": {"nodes": [], "edges": []}},
        "summary": {"association": "", "causality": "", "validation": "", "overall": ""},
        "ground_truth": None, "warnings": [], "error": None,
    }


def _stage(rep, sid, status, message=None):
    for s in rep["stages"]:
        if s["id"] == sid:
            s["status"] = status
            if message:
                s["message"] = message


def _graph(rep):
    ca = rep["causality"]
    res = {t["id"]: t for t in ca["tests"]}
    v = ca["causal_verdict"]
    xy = {"evidence_consistent": "consistent_with_evidence", "contradicted": "contradicted"}.get(v, "not_established")
    gr = res["granger"]["metrics"]
    yx = "not_supported" if gr["p_y_to_x"] >= C.ALPHA else "possible"
    conf = ("supported_by_test" if res["confounder_control"]["result"] == "fail"
            else "not_supported" if res["confounder_control"]["result"] == "pass" else "hypothesised")
    nodes = [{"id": "x", "label": rep["correlation"]["x_label"], "observed": True},
             {"id": "y", "label": rep["correlation"]["y_label"], "observed": True},
             {"id": "factor", "label": rep["provenance"]["factor_label"], "observed": True}]
    edges = [{"source": "x", "target": "y", "status": xy, "evidence": ["lead_lag", "confounder_control", "granger", "placebo_shift"]},
             {"source": "y", "target": "x", "status": yx, "evidence": ["granger", "lead_lag"]},
             {"source": "factor", "target": "x", "status": conf, "evidence": ["confounder_control"]},
             {"source": "factor", "target": "y", "status": conf, "evidence": ["confounder_control"]}]
    return {"nodes": nodes, "edges": edges}


def _summary(rep):
    c, ca, v = rep["correlation"], rep["causality"], rep["validation"]
    ci = c["ci_95"]
    assoc = (f"{c['x_label']} vs {c['y_label']}: Pearson r = {c['r']:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f}, "
             f"n = {c['n']}, HAC p = {c['p_value_hac']:.3g}). Association: {c['association_verdict']}.")
    caus = f"Causal verdict: {ca['causal_verdict']}. {ca['verdict_rule']}"
    n_pass = sum(t["result"] == "pass" for t in v["tests"])
    val = (f"Reliability verdict: {v['reliability_verdict']} ({n_pass}/{len(v['tests'])} tests passed). "
           f"{v['verdict_rule']}")
    overall = (f"Association {c['association_verdict']}; causation {ca['causal_verdict'].replace('_', ' ')}; "
               f"reliability {v['reliability_verdict']}. An association alone never establishes causation.")
    return {"association": assoc, "causality": caus, "validation": val, "overall": overall}


def _ground_truth(rep, preset_id):
    gt = dict(datasets.GROUND_TRUTH[preset_id])
    res = {t["id"]: t["result"] for t in rep["causality"]["tests"]}
    detected = rep["correlation"]["association_verdict"] == "detected"
    gt["system_identified_confounding"] = bool(detected and res["confounder_control"] == "fail")
    gt["verdict_matches_expected"] = rep["causality"]["causal_verdict"] == gt["expected_causal_verdict"]
    gt["note"] = "Revealed only after all tests ran; never given to the hypothesis generator."
    return gt


def run_pipeline(store, run_id, preset_id, reasoner, timeout_s=300, seed=C.SEED):
    rep = store.get(run_id)
    t0 = time.time()
    current = None

    def check():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"run exceeded {timeout_s}s")

    try:
        rep["status"] = "running"
        p = PRESETS[preset_id]
        df = datasets.generate(preset_id, seed=seed)
        df, dropped = S.align(df["x"], df["y"], df["factor"])
        df.columns = ["x", "y", "factor"]
        prov = rep["provenance"]
        prov.update(seed=seed, observations=len(df), sample_start=str(df.index[0].date()),
                    sample_end=str(df.index[-1].date()), missing_data={"rows_dropped": int(dropped)})

        # ---- 1 correlation
        current = "correlation"
        rep["current_stage"] = current
        _stage(rep, current, "running"); store.save(rep)
        rep["correlation"] = correlation.run_correlation(df, p["x_label"], p["y_label"])
        rep["visualization_data"].update(correlation.build_viz_series(df))
        _stage(rep, current, "completed", "Calculated Pearson correlation"); store.save(rep)
        check()

        # ---- 2 causality (hypotheses are blind to ground truth)
        current = "causality"
        rep["current_stage"] = current
        _stage(rep, current, "running"); store.save(rep)
        ctx = {"x_label": p["x_label"], "y_label": p["y_label"],
               "r": rep["correlation"]["r"], "n": rep["correlation"]["n"]}
        hyp, src, warns = reasoner.hypotheses(ctx)
        rep["warnings"] += warns
        rep["causality"] = causal.run_causality(df, rep["correlation"], hyp)
        rep["visualization_data"]["lag_profile"] = rep["causality"]["lag_profile"]
        rep["visualization_data"]["causal_graph"] = _graph(rep)
        _stage(rep, current, "completed", "Reviewed confounders and lead/lag tests"); store.save(rep)
        check()

        # ---- 3 validation
        current = "validation"
        rep["current_stage"] = current
        _stage(rep, current, "running"); store.save(rep)
        rep["validation"] = validation.run_validation(df)
        _stage(rep, current, "completed", "Finished robustness checks"); store.save(rep)
        check()

        # ---- finish: narrative (numbers already computed), summary, ground truth
        facts = to_jsonable({"correlation": {k: rep["correlation"][k] for k in
                                             ("x_label", "y_label", "r", "n", "association_verdict")},
                             "causality": {k: rep["causality"][k] for k in ("causal_verdict", "verdict_rule")},
                             "validation": {"reliability_verdict": rep["validation"]["reliability_verdict"],
                                            "verdict_rule": rep["validation"]["verdict_rule"],
                                            "tests": [{"id": t["id"], "result": t["result"]} for t in rep["validation"]["tests"]]}})
        nar, nsrc, nwarn = reasoner.narrative(facts)
        rep["warnings"] += nwarn
        rep["causality"]["reasoning"] = nar["causality_reasoning"]
        rep["validation"]["reasoning"] = nar["validation_reasoning"]
        prov["reasoning_source"] = src if src == nsrc else (
            "template_fallback" if "template_fallback" in (src, nsrc) else "gemini")
        rep["summary"] = _summary(rep)
        rep["ground_truth"] = _ground_truth(rep, preset_id)
        if not rep["correlation"]["checks"]["stationary_at_5pct"]:
            rep["warnings"].append("ADF test did not reject a unit root for at least one series.")
        rep["warnings"].append("Synthetic data: results demonstrate the method, not a real-market finding.")
        rep["status"], rep["current_stage"] = "completed", "complete"
        store.save(rep)
    except Exception as e:
        traceback.print_exc()
        rep = store.get(run_id) or rep
        rep["status"] = "failed"
        if current:
            _stage(rep, current, "failed")
        rep["error"] = {"code": "timeout" if isinstance(e, TimeoutError) else "internal_error",
                        "stage": current, "message": f"{type(e).__name__}: {e}"}
        store.save(rep)
