"""Stage 2: could the association be causal? Tests are necessary-not-sufficient."""
import warnings
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from . import criteria as C, stats as S


def _test(id_, name, metrics, threshold, result, explanation, limitations):
    return {"id": id_, "name": name, "metrics": metrics, "threshold": threshold,
            "result": result, "explanation": explanation, "limitations": limitations}


def _prewhiten(x, y, order=C.LAG_RANGE):
    """Box-Jenkins prewhitening: filter BOTH series with the AR(order) filter
    fitted to X, so X's own autocorrelation cannot masquerade as Y-leads-X."""
    X = np.column_stack([x[order - i - 1: len(x) - i - 1] for i in range(order)])
    phi = np.linalg.lstsq(X, x[order:], rcond=None)[0]

    def filt(s):
        return s[order:] - np.column_stack([s[order - i - 1: len(s) - i - 1] for i in range(order)]) @ phi
    return filt(x), filt(y)


def _lead_lag(x, y):
    x, y = _prewhiten(x, y)
    ks = list(range(-C.LAG_RANGE, C.LAG_RANGE + 1))
    rs, ps = [], []
    for k in ks:
        xa, ya = S.lagged_pair(x, y, k)
        r, p = S.hac_corr(xa, ya)
        rs.append(r)
        ps.append(p)
    adj, rej = S.holm(ps, C.ALPHA)
    profile = [{"lag": k, "r": r, "p_hac": p, "p_holm": a, "significant": bool(s)}
               for k, r, p, a, s in zip(ks, rs, ps, adj, rej)]
    x_lead = [e for e in profile if e["lag"] >= 1 and e["significant"]]
    y_lead = [e for e in profile if e["lag"] <= -1 and e["significant"]]
    if x_lead and not y_lead:
        res, ex = "pass", "Only X-leads-Y lags are significant after Holm correction: time order is consistent with X->Y."
    elif y_lead and not x_lead:
        res, ex = "fail", "Only Y-leads-X lags are significant: time order points the opposite way (reverse causation)."
    elif x_lead and y_lead:
        res, ex = "unknown", "Both directions are significant: feedback or a shared persistent driver."
    else:
        res, ex = "fail", "No lead/lag is significant after Holm correction: no evidence that X precedes Y."
    return profile, _test("lead_lag", "Lead/lag scan on prewhitened series (k=-5..+5, Holm-corrected)",
                          {"n_tests": len(ks), "x_leads_significant": [e["lag"] for e in x_lead],
                           "y_leads_significant": [e["lag"] for e in y_lead]},
                          f"Holm p<{C.ALPHA}", res, ex,
                          ["Both series are filtered with X's AR(5) filter (prewhitening) so autocorrelation does not fake lead/lag.",
                           "Temporal precedence is necessary for causation but not sufficient.",
                           "Contemporaneous (k=0) effects cannot be ordered with daily data."])


def _best_lag(x, y):
    return max(range(0, C.LAG_RANGE + 1),
               key=lambda k: abs(np.corrcoef(*S.lagged_pair(x, y, k))[0, 1]))


def _confounder_control(x, y, f, k):
    xa, ya = S.lagged_pair(x, y, k)
    _, fa = S.lagged_pair(x, f, k)           # factor at time t, aligned with y_t
    kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": S.HAC_LAGS}}
    b_raw = sm.OLS(ya, sm.add_constant(xa)).fit(**kw)
    b_ctl = sm.OLS(ya, sm.add_constant(np.column_stack([xa, fa]))).fit(**kw)
    raw, ctl, p_ctl = float(b_raw.params[1]), float(b_ctl.params[1]), float(b_ctl.pvalues[1])
    att = float(1 - abs(ctl) / abs(raw)) if raw != 0 else None
    m = {"lag": k, "beta_raw": raw, "beta_controlled": ctl, "p_controlled": p_ctl,
         "attenuation": att, "p_raw": float(b_raw.pvalues[1])}
    if b_raw.pvalues[1] >= C.ALPHA:
        res, ex = "unknown", "No raw association at this lag, so there is nothing for a confounder to explain."
    elif (att is not None and att >= C.CONFOUND_ATTENUATION) or p_ctl >= C.ALPHA:
        res, ex = "fail", (f"Controlling for the common market factor shrinks the X coefficient by "
                           f"{att:.0%} (p={p_ctl:.3g}): the association is largely explained by the common factor.")
    else:
        res, ex = "pass", f"X coefficient survives control for the market factor (attenuation {att:.0%}, p={p_ctl:.3g})."
    return _test("confounder_control", "Regression control for observed common factor", m,
                 f"controlled p<{C.ALPHA} and attenuation<{C.CONFOUND_ATTENUATION:.0%}", res, ex,
                 ["Only controls for the factor we observe; unobserved confounders remain possible.",
                  "Regression adjustment is not proof of causality."])


def _granger(x, y):
    out = {}
    for name, data in (("x_to_y", np.column_stack([y, x])), ("y_to_x", np.column_stack([x, y]))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=C.GRANGER_MAXLAG, verbose=False)
        ps = [res[l][0]["ssr_ftest"][1] for l in range(1, C.GRANGER_MAXLAG + 1)]
        out[name] = float(min(1.0, min(ps) * C.GRANGER_MAXLAG))   # Bonferroni over lags
    sx, sy = out["x_to_y"] < C.ALPHA, out["y_to_x"] < C.ALPHA
    if sx and not sy:
        res_, ex = "pass", "X predicts Y beyond Y's own history, but not the reverse."
    elif sx and sy:
        res_, ex = "unknown", "Prediction runs in both directions."
    else:
        res_, ex = "fail", "X does not add predictive information about Y beyond Y's own past."
    return _test("granger", "Granger-style predictive test (lags 1-3, Bonferroni)",
                 {"p_x_to_y": out["x_to_y"], "p_y_to_x": out["y_to_x"]}, f"p<{C.ALPHA}", res_, ex,
                 ["Predictive precedence, not causation: a common driver can create Granger 'causality'.",
                  "Assumes linear dynamics and stationarity."])


def _placebo(x, y, k):
    p = S.circular_shift_placebo(x, y, k, C.PLACEBO_DRAWS, seed=C.SEED)
    ok = p < C.ALPHA
    return _test("placebo_shift", "Circular-shift placebo (negative control)", {"lag": k, "p_placebo": p},
                 f"p<{C.ALPHA}", "pass" if ok else "fail",
                 "Association is stronger than for time-shifted copies of X (not a trend/autocorrelation artefact)."
                 if ok else "Time-shifted placebo copies of X produce comparably strong associations.",
                 ["Rules out spurious co-trending only. A shared factor still passes this test."])


def run_causality(df, corr, explanations):
    x, y, f = df["x"].to_numpy(), df["y"].to_numpy(), df["factor"].to_numpy()
    k = _best_lag(x, y)
    profile, t_ll = _lead_lag(x, y)
    t_cc = _confounder_control(x, y, f, k)
    t_gr = _granger(x, y)
    t_pl = _placebo(x, y, k)
    tests = [t_ll, t_cc, t_gr, t_pl]
    r = {t["id"]: t["result"] for t in tests}

    detected = corr["association_verdict"] == "detected"
    llm = t_ll["metrics"]
    reverse_only = bool(llm["y_leads_significant"]) and not llm["x_leads_significant"]
    confounded = r["confounder_control"] == "fail" and detected
    if not detected:
        verdict, why = "not_established", "No robust association was detected, so there is no relationship to explain."
    elif confounded or reverse_only:
        verdict = "contradicted"
        why = ("Controlling for the observed common factor removes most of the association." if confounded
               else "Time ordering points from Y to X.")
    elif all(r[i] == "pass" for i in ("lead_lag", "confounder_control", "granger", "placebo_shift")):
        verdict, why = "evidence_consistent", ("All four checks are consistent with X->Y; this is not proof: "
                                               "unobserved confounders and model misspecification remain possible.")
    else:
        verdict, why = "not_established", "Some checks did not support a causal reading and none contradicted it decisively."

    return {
        "candidate_explanations": explanations["candidate_explanations"],
        "possible_confounders": explanations["possible_confounders"],
        "tests": tests, "selected_lag": k,
        "causal_verdict": verdict, "verdict_rule": why,
        "verdict_criteria": {
            "contradicted": "association detected AND (confounder control fails OR only Y-leads-X lags significant)",
            "evidence_consistent": "association detected AND lead_lag, confounder_control, granger, placebo_shift all pass",
            "not_established": "everything else (default). 'proven' is never used."},
        "reasoning": "",
        "lag_profile": profile,
    }
