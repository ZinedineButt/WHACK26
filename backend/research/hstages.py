"""Per-hypothesis stages on real (or fixture) series.

Sample discipline: stages 1-2 (exploratory) see only the first 75% of the
sample; the final 25% (holdout) is touched once, in stage 3."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import criteria as C, registry as R, stats as S, strength as ST, transforms as T, validation as V
from .causal import _granger, _lead_lag, _placebo, _test

MIN_ROWS = 250

# Scheduled FOMC decision days (public Fed calendar). Used only by the event-study check on LIVE data.
FOMC_DATES = [
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]
NEG_CONTROL_CANDIDATES = ("DHHNGSP", "DEXJPUS", "DEXUSEU")     # unrelated to most equity / volatility outcomes
FAMILY_LAGS = list(range(1, 6))


def hac_lags_for(h):
    w = R.FORWARD_OUTCOMES.get(h["outcome"]["transform"], 0)
    return max(S.HAC_LAGS, 2 * w)


def is_forward(h):
    return h["outcome"]["transform"] in R.FORWARD_OUTCOMES


def build_frame(h, levels, controls):
    """levels: {variable_id: level Series}; controls: {variable_id: transform}. Aligns LEVELS on common dates first,
    then transforms, so no transform mixes information across mismatched grids."""
    pid, oid = h["predictor"]["variable_id"], h["outcome"]["variable_id"]
    ids = sorted({pid, oid, *controls})
    aligned = pd.concat({i: levels[i] for i in ids}, axis=1, join="inner").dropna()
    if len(aligned) < MIN_ROWS:
        raise ValueError(f"only {len(aligned)} overlapping observations (need {MIN_ROWS})")
    df = pd.DataFrame({
        "x": T.apply(aligned[pid], h["predictor"]["transform"]),
        "y": T.apply(aligned[oid], h["outcome"]["transform"]),
    })
    for c, tr in controls.items():
        df[f"c_{c}"] = T.apply(aligned[c], tr)
    if is_forward(h):
        df["c_trailing_vol"] = T.apply(aligned[oid], R.TRAILING_LOGVOL_5)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < MIN_ROWS:
        raise ValueError(f"only {len(df)} usable rows after transforms (need {MIN_ROWS})")
    return df, {"raw_rows_aligned": int(len(aligned)), "usable_rows": int(len(df)),
                "rows_lost_to_alignment_or_transform": int(len(aligned) - len(df))}


def split_frame(df):
    a, b = V.split_indices(len(df))
    return df.iloc[:b], df.iloc[b:]


# ------------------------------------------------------------------ stage 1
def stage1(dev, h):
    k, L = h["lag_days"], hac_lags_for(h)
    x, y = dev["x"].to_numpy(), dev["y"].to_numpy()
    xa, ya = S.lagged_pair(x, y, k)
    r, p_iid = S.pearson(xa, ya)
    _, p_hac = S.hac_corr(xa, ya, L)
    rho, p_rho = S.spearman(xa, ya)
    n = len(xa)
    fam_lags = [0] if is_forward(h) else FAMILY_LAGS
    fam = []
    for kk in fam_lags:
        a, b = S.lagged_pair(x, y, kk)
        rr, pp = S.hac_corr(a, b, L)
        fam.append({"lag": kk, "r": rr, "p_hac": pp})
    adj, _ = S.holm([f["p_hac"] for f in fam], C.ALPHA)
    for f, a in zip(fam, adj):
        f["p_holm"] = a
    r0 = float(np.corrcoef(x, y)[0, 1])  # same-time association, descriptive only

    sign_ok = h["direction"] == "any" or (np.sign(r) == (1 if h["direction"] == "positive" else -1))
    if n < C.MIN_N:
        verdict, why = "inconclusive", f"n={n} < {C.MIN_N}."
    elif p_hac < C.ASSOC_DETECT_P and abs(r) >= C.ASSOC_DETECT_ABS_R and np.sign(rho) == np.sign(r):
        if sign_ok:
            verdict, why = "detected", (f"HAC p={p_hac:.3g} < {C.ASSOC_DETECT_P}, |r|={abs(r):.3f} >= "
                                        f"{C.ASSOC_DETECT_ABS_R}, Spearman agrees, sign matches pre-registered direction.")
        else:
            verdict, why = "inconclusive", (f"Significant (HAC p={p_hac:.3g}) but the sign is OPPOSITE to the "
                                            f"pre-registered direction '{h['direction']}'.")
    elif p_hac >= C.ASSOC_NULL_P:
        verdict, why = "not_detected", f"HAC p={p_hac:.3g} >= {C.ASSOC_NULL_P}."
    else:
        verdict, why = "inconclusive", "Evidence is between the detection and null criteria."
    ci = S.fisher_ci(r, n)
    warn = []
    adf_x, adf_y = S.adf_pvalue(x), S.adf_pvalue(y)
    if adf_x >= 0.05 or adf_y >= 0.05:
        warn.append(f"ADF does not reject a unit root (p_x={adf_x:.3f}, p_y={adf_y:.3f}); stationarity doubtful.")
    if is_forward(h):
        warn.append("Forward-window outcome: observations overlap, so HAC lags were widened to %d." % L)
    return {
        "strength": ST.association_strength({"r": r, "p_value_hac": p_hac, "direction_consistent": bool(sign_ok)}),
        "x_label": f"{h['predictor']['label']} [{h['predictor']['transform']}]",
        "y_label": f"{h['outcome']['label']} [{h['outcome']['transform']}]",
        "method": "pearson", "primary_lag": k,
        "timing": ("predictor stamped t vs outcome over t+1..t+%d (forward window)" % R.FORWARD_OUTCOMES[h["outcome"]["transform"]]
                   if is_forward(h) else f"predictor at day t vs outcome at day t+{k}"),
        "r": r, "p_value": p_iid, "p_value_hac": p_hac, "hac_maxlags": L, "ci_95": ci, "n": n,
        "effect_size_r2": r * r, "spearman_rho": rho, "spearman_p": p_rho,
        "same_time_r": r0, "direction_preregistered": h["direction"], "direction_consistent": bool(sign_ok),
        "lag_family": fam, "lag_family_correction": "holm",
        "association_verdict": verdict, "verdict_rule": why,
        "sample": {"start": str(dev.index[0].date()), "end": str(dev.index[-1].date()), "n_rows": int(len(dev)),
                   "label": "development sample (first 75%); holdout sealed"},
        "criteria": {"detected": f"HAC p<{C.ASSOC_DETECT_P}, |r|>={C.ASSOC_DETECT_ABS_R}, Spearman same sign, sign matches direction",
                     "not_detected": f"HAC p>={C.ASSOC_NULL_P}"},
        "assumptions": ["Pearson r is linear association only.", "p_value assumes i.i.d.; p_value_hac (Newey-West) drives the verdict.",
                        "Only pre-registered lags 1..5 were tested; Holm-corrected across them."],
        "checks": {"adf_p_x": adf_x, "adf_p_y": adf_y}, "warnings": warn,
    }


# ------------------------------------------------------------------ stage 2
def _control_regression(dev, h):
    k, L = h["lag_days"], hac_lags_for(h)
    x, y = dev["x"].to_numpy(), dev["y"].to_numpy()
    ctl_cols = [c for c in dev.columns if c.startswith("c_")]
    if not ctl_cols:
        return None, _test("confounder_control", "Regression control for observed common factors", {},
                           "n/a", "not_run", "No control variables available.", ["Cannot test confounding without controls."])
    xa, ya = S.lagged_pair(x, y, k)
    n = len(xa)
    C_ = np.column_stack([dev[c].to_numpy()[: len(dev) - k] for c in ctl_cols])   # controls dated t (pre-outcome)
    kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": L}}
    b_raw = sm.OLS(ya, sm.add_constant(xa)).fit(**kw)
    b_ctl = sm.OLS(ya, sm.add_constant(np.column_stack([xa, C_]))).fit(**kw)
    raw, ctl, p_ctl = float(b_raw.params[1]), float(b_ctl.params[1]), float(b_ctl.pvalues[1])
    att = float(1 - abs(ctl) / abs(raw)) if raw else None
    m = {"lag": k, "controls": ctl_cols, "beta_raw": raw, "beta_controlled": ctl, "p_raw": float(b_raw.pvalues[1]),
         "p_controlled": p_ctl, "attenuation": att, "n": n}
    if b_raw.pvalues[1] >= C.ALPHA:
        res, ex = "not_run", "No raw association at the primary lag, so there is nothing for a confounder to explain."
    elif (att is not None and att >= C.CONFOUND_ATTENUATION) or p_ctl >= C.ALPHA:
        res, ex = "fail", (f"Controlling for {', '.join(ctl_cols)} (dated at the predictor time, i.e. pre-outcome) shrinks the "
                           f"predictor coefficient by {att:.0%} (p={p_ctl:.3g}).")
    else:
        res, ex = "pass", f"Predictor coefficient survives the controls (attenuation {att:.0%}, p={p_ctl:.3g})."
    return m, _test("confounder_control", "Regression control for observed common factors", m,
                    f"controlled p<{C.ALPHA} and attenuation<{C.CONFOUND_ATTENUATION:.0%}", res, ex,
                    ["Only observed controls; unobserved confounders remain possible.",
                     "Controls are dated at the predictor time to avoid conditioning on post-treatment variables."])


def _normalise(t, not_run_unknown=False):
    if t["result"] == "unknown":
        t["result"] = "not_run" if not_run_unknown else "inconclusive"
    return t


def _forward_granger(dev, h):
    """Volatility outcomes: does the predictor add information beyond the outcome's OWN recent volatility?"""
    k, L = h["lag_days"], hac_lags_for(h)
    x, y, tv = dev["x"].to_numpy(), dev["y"].to_numpy(), dev["c_trailing_vol"].to_numpy()
    xa, ya = S.lagged_pair(x, y, k)
    tva = tv[: len(tv) - k]
    fit = sm.OLS(ya, sm.add_constant(np.column_stack([tva, xa]))).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    p, beta = float(fit.pvalues[2]), float(fit.params[2])
    ok = p < C.ALPHA
    return _test("granger", "Granger-style predictive test (vs the outcome's own recent volatility)",
                 {"p_predictor": p, "beta_predictor": beta, "n": len(xa)}, f"p<{C.ALPHA}", "pass" if ok else "fail",
                 "X still helps predict future volatility after allowing for Y's own recent volatility."
                 if ok else "X adds nothing beyond Y's own recent volatility: it may just mirror what has already happened.",
                 ["Linear model with the outcome's trailing 5-day volatility as the only history term.",
                  "Predictive precedence, not causation."])


def _forward_timing(dev, h):
    """Volatility outcomes: does the predictor line up more with the volatility that FOLLOWS than with what came BEFORE?"""
    k, L = h["lag_days"], hac_lags_for(h)
    x, y = dev["x"].to_numpy(), dev["y"].to_numpy()
    past = dev["c_trailing_vol"].shift(1).to_numpy()          # volatility over the days ending t-1: already known
    ok_ = ~np.isnan(past)
    r_f, p_f = S.hac_corr(*S.lagged_pair(x, y, k), L)
    r_p, p_p = S.hac_corr(x[ok_], past[ok_], L)
    m = {"r_future": r_f, "p_future": p_f, "r_past": r_p, "p_past": p_p}
    if p_f >= C.ALPHA:
        res, ex = "fail", "No significant link with the volatility that follows, so there is no timing to check."
    elif abs(r_f) > abs(r_p):
        res, ex = "pass", (f"X lines up more with the volatility that FOLLOWS (|r|={abs(r_f):.2f}) than with the volatility that came BEFORE "
                           f"(|r|={abs(r_p):.2f}): consistent with X leading Y.")
    else:
        res, ex = "inconclusive", (f"X lines up as much with volatility that already happened (|r|={abs(r_p):.2f}) as with what follows "
                                   f"(|r|={abs(r_f):.2f}): it may mostly reflect recent volatility rather than lead it.")
    return _test("lead_lag", "Timing check (future vs past volatility)", m, "|r| with the future window > |r| with the past window", res, ex,
                 ["Compares the predictor with the 5 days after t and the 5 days ending t-1.",
                  "Temporal precedence is necessary for causation but not sufficient."])


def _negative_control(dev, h, extras):
    if not extras or extras.get("neg_series") is None:
        why = (extras or {}).get("neg_note") or "No negative-control series is available for this run."
        return _test("negative_control", "Negative-control outcome", {}, "n/a", "not_run", why, [])
    k, L = h["lag_days"], hac_lags_for(h)
    neg = extras["neg_series"].reindex(dev.index).to_numpy()
    ok_ = ~np.isnan(neg)
    x = dev["x"].to_numpy()[ok_]
    z = neg[ok_]
    if len(x) < C.MIN_N + k:
        return _test("negative_control", "Negative-control outcome", {}, "n/a", "not_run", "Too few overlapping days with the control series.", [])
    r, p = S.hac_corr(*S.lagged_pair(x, z, k), L)
    quiet = p >= C.ALPHA or abs(r) < C.ASSOC_DETECT_ABS_R
    lab = extras["neg_label"]
    return _test("negative_control", f"Negative-control outcome ({lab})", {"r": r, "p_hac": p, "control": extras["neg_id"], "n": int(len(x))},
                 f"no link: p>={C.ALPHA} or |r|<{C.ASSOC_DETECT_ABS_R}", "pass" if quiet else "fail",
                 f"The predictor shows no link with {lab}, which it has no reason to affect (r={r:.2f}, p={p:.3g})."
                 if quiet else f"The predictor ALSO shows a link with {lab} (r={r:.2f}, p={p:.3g}), which it should not affect: the pattern may be generic, not specific.",
                 ["A control series is only 'unrelated' by judgement; it is fixed by the program, not chosen by the AI.",
                  "Passing means the pattern is specific to the outcome, not that it is causal."])


def _event_study(dev, h, extras):
    if not extras or not extras.get("live"):
        return _test("event_study", "Event study", {}, "n/a", "not_run", "No real announcement calendar applies to synthetic dates.", [])
    k, L = h["lag_days"], hac_lags_for(h)
    near = set()
    for d in pd.to_datetime(FOMC_DATES):
        for off in (-1, 0, 1):
            near.add(d + pd.Timedelta(days=off))
    keep = ~dev.index.isin(near)
    n_ev = int((~keep).sum())
    if n_ev < 5:
        return _test("event_study", "Event study (Fed meeting days)", {"event_days_in_sample": n_ev}, "n/a", "not_run",
                     "Fewer than 5 Fed-meeting days fall inside this sample.", [])
    x, y = dev["x"].to_numpy(), dev["y"].to_numpy()
    xa, ya = S.lagged_pair(x, y, k)
    m = keep[: len(xa)]
    r_all, _ = S.hac_corr(xa, ya, L)
    r_ex, p_ex = S.hac_corr(xa[m], ya[m], L)
    ok = np.sign(r_ex) == np.sign(r_all) and abs(r_ex) >= 0.75 * abs(r_all) and p_ex < C.ALPHA
    return _test("event_study", "Event study (Fed meeting days removed)", {"event_days_removed": n_ev, "r_all": r_all, "r_without_events": r_ex, "p_without_events": p_ex},
                 "same sign, |r| keeps >=75% of its size, still p<0.05", "pass" if ok else "fail",
                 f"The link survives removing the {n_ev} days around Fed meetings (r {r_all:.2f} to {r_ex:.2f}): it is not driven by announcement days."
                 if ok else f"Removing the {n_ev} days around Fed meetings changes the link (r {r_all:.2f} to {r_ex:.2f}): announcement days may be driving it.",
                 ["Uses scheduled FOMC decision days from the public Fed calendar, plus one day either side.",
                  "Rows are dropped, not interpolated, so HAC lags are slightly approximate."])


def stage2(dev, h, proposal, corr, extras=None):
    k = h["lag_days"]
    x, y = dev["x"].to_numpy(), dev["y"].to_numpy()
    fwd = is_forward(h)
    tests = []
    if fwd:
        profile = []
        tests.append(_normalise(_forward_timing(dev, h)))
        tests.append(_normalise(_forward_granger(dev, h)))
    else:
        profile, t_ll = _lead_lag(x, y)
        tests.append(_normalise(t_ll))
        tests.append(_normalise(_granger(x, y)))
    _, t_cc = _control_regression(dev, h)
    tests.insert(1, _normalise(t_cc, not_run_unknown=True))
    tests.append(_normalise(_placebo(x, y, k)))
    tests.append(_negative_control(dev, h, extras))
    tests.append(_event_study(dev, h, extras))
    r = {t["id"]: t["result"] for t in tests}
    ll = next(t for t in tests if t["id"] == "lead_lag")["metrics"]
    cs = ST.causal_strength(tests)
    detected = corr["association_verdict"] == "detected"
    reverse_only = bool(ll.get("y_leads_significant")) and not ll.get("x_leads_significant")
    confounded = detected and r["confounder_control"] == "fail"
    if not detected:
        verdict, why = "not_established", "No robust association was detected, so there is nothing to explain causally."
    elif confounded or reverse_only:
        verdict = "contradicted"
        why = ("Observed controls remove most of the association." if confounded else "Time ordering points from outcome to predictor.")
    elif cs["pct"] is not None and cs["pct"] >= ST.CAUSAL_PLAUSIBLE_MIN and r["confounder_control"] == "pass":
        verdict, why = "evidence_consistent", (f"{cs['checks_passed']} of {cs['checks_run']} causal checks passed (strength {cs['pct']}/100, {cs['label']}) "
                                               "and the control for other explanations held. This is NOT proof of causation.")
    else:
        verdict, why = "not_established", "Not all required checks passed or could be run; causal reading not established."
    return {
        "mechanisms": proposal["mechanisms"], "alternative_explanations": proposal["alternative_explanations"],
        "possible_confounders": proposal["confounders"], "assumptions": proposal["assumptions"],
        "tests": tests, "causal_verdict": verdict, "verdict_rule": why, "strength": cs,
        "verdict_criteria": {
            "contradicted": "association detected AND (control regression fails OR only outcome-leads-predictor lags significant)",
            "evidence_consistent": "association detected AND weighted causal score >= 75 AND the control for other explanations passes",
            "not_established": "everything else (default). 'proven' is never used."},
        "reasoning": "", "lag_profile": profile, "reasoning_source": proposal.get("source"),
    }


def causal_graph(h, causality, proposal):
    nodes = [{"id": h["predictor"]["variable_id"], "label": h["predictor"]["label"], "role": "predictor", "observed": True},
             {"id": h["outcome"]["variable_id"], "label": h["outcome"]["label"], "role": "outcome", "observed": True}]
    for c in h["control_variables"] + [x for x in h.get("market_control", []) if x not in h["control_variables"]]:
        nodes.append({"id": c, "label": c, "role": "control", "observed": True})
    r = {t["id"]: t["result"] for t in causality["tests"]}
    v = causality["causal_verdict"]
    xy = {"evidence_consistent": "consistent_with_evidence", "contradicted": "contradicted"}.get(v, "not_established")
    pid, oid = nodes[0]["id"], nodes[1]["id"]
    edges = [{"source": pid, "target": oid, "status": xy, "kind": "hypothesised_causal", "evidence": ["lead_lag", "confounder_control", "granger", "placebo_shift"]}]
    for n in nodes[2:]:
        st = "supported_by_test" if r["confounder_control"] == "fail" else "not_supported" if r["confounder_control"] == "pass" else "hypothesised"
        edges += [{"source": n["id"], "target": pid, "status": st, "kind": "possible_confounder", "evidence": ["confounder_control"]},
                  {"source": n["id"], "target": oid, "status": st, "kind": "possible_confounder", "evidence": ["confounder_control"]}]
    ids = {n["id"] for n in nodes}
    for e in proposal.get("dag_edges", []):
        for end in (e["source"], e["target"]):
            if end not in ids:
                nodes.append({"id": end, "label": end, "role": "proposed", "observed": not end.startswith("latent:")})
                ids.add(end)
        if not any(x["source"] == e["source"] and x["target"] == e["target"] for x in edges):
            edges.append({"source": e["source"], "target": e["target"], "status": "proposed_untested",
                          "kind": "model_proposed", "evidence": [], "rationale": e.get("rationale", "")})
    return {"nodes": nodes, "edges": edges}


# ------------------------------------------------------------------ stage 3
def stage3(df, h):
    return V.run_validation(df, fixed_lag=h["lag_days"], lag_family=[0] if is_forward(h) else FAMILY_LAGS,
                            hac_lags=hac_lags_for(h), development_only=True)


def _fit(xa, ya, n_dev):
    """Least-squares line over the development pairs, for drawing only (r comes from stage 1)."""
    n_dev = max(2, min(n_dev, len(xa)))
    slope, icpt = np.polyfit(xa[:n_dev], ya[:n_dev], 1)
    return {"slope": float(slope), "intercept": float(icpt), "x_min": float(xa.min()), "x_max": float(xa.max()),
            "fitted_on": "development sample"}


def viz(df, h, corr, val, window=C.ROLLING_WINDOW):
    k = h["lag_days"]
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    a, b = V.split_indices(len(df))
    dates = [d.date().isoformat() for d in df.index]
    xa, ya = S.lagged_pair(x, y, k)
    xs = x[: len(x) - k]
    rr = S.rolling_r(xa, ya, window)
    sample = lambda i: "development" if i < b else "holdout"
    ctl = [c for c in df.columns if c.startswith("c_")]
    out = {
        "timeseries": [dict({"date": d, "predictor": float(p), "outcome": float(o), "sample": sample(i)},
                            **{c: float(df[c].iloc[i]) for c in ctl}) for i, (d, p, o) in enumerate(zip(dates, x, y))],
        "scatter": [{"x": float(p), "y": float(o), "date": dates[i + k], "sample": sample(i)} for i, (p, o) in enumerate(zip(xa, ya))],
        "rolling_r": [{"date": dates[i + k], "r": r} for i, r in enumerate(rr)],
        "rolling_window": window, "scatter_lag": k,
        "scatter_fit": _fit(xa, ya, b - k),
        "lag_profile": (corr["lag_family"]), "lead_lag_profile": [],
        "validation_metrics": {},
    }
    if val and val.get("tests"):
        m = {t["id"]: t["metrics"] for t in val["tests"]}
        out["validation_metrics"] = {
            "window_r": m.get("window_stability", {}).get("window_r", []),
            "r_by_split": {"train": m["validation_replication"]["r_train"], "validation": m["validation_replication"]["r_validation"],
                           "holdout": m["holdout_single_evaluation"]["r_holdout"]},
            "bootstrap_ci_95": m["block_bootstrap"]["ci_95"], "split": val.get("split")}
    return out
