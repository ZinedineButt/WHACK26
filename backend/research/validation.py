"""Stage 3: would the claim survive scrutiny?

Claim under test: "X at the lag selected on the TRAIN split is associated with Y,
and the association is stable out-of-sample". This is an observational /
predictive reliability claim, NOT a causal one.

No-leakage protocol: the lag is chosen using the train split only; validation is
used to check replication; the holdout is evaluated exactly once with all
parameters frozen.
"""
import numpy as np
from . import criteria as C, stats as S


def _t(id_, name, metrics, threshold, result, explanation, limitations, critical=False):
    return {"id": id_, "name": name, "metrics": metrics, "threshold": threshold, "result": result,
            "explanation": explanation, "limitations": limitations, "critical": critical}


def _pair(df_slice, k):
    return S.lagged_pair(df_slice["x"].to_numpy(), df_slice["y"].to_numpy(), k)


from . import strength as _ST  # noqa: E402


def verdict_from_tests(tests):
    """Graded reliability verdict from the weighted score (see strength.py). One wobbly check no longer sinks an otherwise
    strong result, but a failed holdout is only forgiven when everything else is strong."""
    from . import strength as ST
    st = ST.reliability_strength(tests)
    pct = st["pct"]
    fails = [t["id"] for t in tests if t["result"] == "fail"]
    hold_fail = any(t["id"] == "holdout_single_evaluation" and t["result"] == "fail" for t in tests)
    if pct is None:
        return "inconclusive", "No test could be run."
    summary = f"{st['checks_passed']} of {st['checks_run']} checks passed (strength {pct}/100, {st['label']})"
    if pct < ST.UNRELIABLE_MAX or (hold_fail and pct < ST.HOLDOUT_FAIL_REJECT_BELOW):
        return "rejected", f"{summary}. Failed: {', '.join(fails) or 'none'}."
    if pct >= ST.RELIABLE_MIN and not hold_fail:
        return "supported", f"{summary}" + (f"; the exception: {', '.join(fails)}." if fails else ".")
    return "inconclusive", f"{summary}. Mixed evidence" + (f"; failed: {', '.join(fails)}." if fails else ".")


def split_indices(n):
    a = int(n * C.SPLIT[0])
    b = int(n * (C.SPLIT[0] + C.SPLIT[1]))
    return a, b


def select_lag_on_train(train):
    """Uses ONLY train data."""
    return max(range(0, C.LAG_RANGE + 1),
               key=lambda k: abs(np.corrcoef(*_pair(train, k))[0, 1]))


def run_validation(df, fixed_lag=None, lag_family=None, hac_lags=S.HAC_LAGS, development_only=False):
    """fixed_lag: pre-registered lag (no selection). lag_family: lags actually
    tried (Holm / placebo family). development_only: robustness diagnostics use
    ONLY train+validation data; the holdout is touched once, by one test."""
    n = len(df)
    if n < C.MIN_VALIDATION_N:
        return {"tests": [], "reliability_verdict": "inconclusive",
                "verdict_rule": f"n={n} < {C.MIN_VALIDATION_N}: too little data to validate.",
                "reasoning": "", "split": None, "claim_tested": None}
    a, b = split_indices(n)
    train, val, hold = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    k = fixed_lag if fixed_lag is not None else select_lag_on_train(train)
    ks = list(lag_family) if lag_family else list(range(0, C.LAG_RANGE + 1))
    dev = df.iloc[:b] if development_only else df    # sample for non-holdout diagnostics
    block = max(C.BOOTSTRAP_BLOCK, 2 * hac_lags)
    tests = []

    xt, yt = _pair(train, k)
    r_tr, _ = S.hac_corr(xt, yt, hac_lags)

    xv, yv = _pair(val, k)
    r_v, p_v = S.hac_corr(xv, yv, hac_lags)
    ok = p_v < C.ALPHA and np.sign(r_v) == np.sign(r_tr)
    tests.append(_t("validation_replication", "Replication on validation split (lag frozen from train)",
                    {"lag": k, "r_train": r_tr, "r_validation": r_v, "p_hac_validation": p_v, "n": len(xv)},
                    f"HAC p<{C.ALPHA} and same sign as train", "pass" if ok else "fail",
                    "Train-selected relationship replicates on later data." if ok
                    else "Train-selected relationship does not replicate on later data.",
                    ["Single split; a short validation window has limited power."], critical=True))

    # Holdout: one evaluation, frozen lag, slope fitted on train+validation only.
    xh, yh = _pair(hold, k)
    r_h, p_h = S.hac_corr(xh, yh, hac_lags)
    tv = df.iloc[:b]
    xf, yf = _pair(tv, k)
    slope, icpt = np.polyfit(xf, yf, 1)
    pred = icpt + slope * xh
    sse = float(((yh - pred) ** 2).sum())
    sst = float(((yh - yf.mean()) ** 2).sum())
    oos_r2 = 1 - sse / sst
    ok = p_h < C.ALPHA and np.sign(r_h) == np.sign(r_tr) and oos_r2 > 0
    tests.append(_t("holdout_single_evaluation", "Final holdout (evaluated once)",
                    {"lag": k, "r_holdout": r_h, "p_hac_holdout": p_h, "oos_r2": float(oos_r2), "n": len(xh)},
                    f"HAC p<{C.ALPHA}, same sign, out-of-sample R2>0", "pass" if ok else "fail",
                    "Frozen relationship holds on untouched final data." if ok
                    else "Frozen relationship fails on the untouched final data.",
                    ["Holdout is used once; repeated looking would invalidate it.",
                     "OOS R2 for lag 0 is explanatory, not forecasting."], critical=True))

    # Full-sample HAC with Holm over every lag attempted (0..LAG_RANGE).
    xd, yd = dev["x"].to_numpy(), dev["y"].to_numpy()
    ps = [S.hac_corr(*S.lagged_pair(xd, yd, kk), hac_lags)[1] for kk in ks]
    adj, _ = S.holm(ps, C.ALPHA)
    ik = ks.index(k)
    p_adj = adj[ik]
    tests.append(_t("hac_holm", "Full-sample HAC inference, Holm-corrected over lags tried",
                    {"lag": k, "p_hac": ps[ik], "p_holm": p_adj, "n_tests_attempted": len(ks), "lags_tried": ks},
                    f"Holm p<{C.ALPHA}", "pass" if p_adj < C.ALPHA else "fail",
                    "Survives Newey-West errors and the lag-selection multiplicity correction."
                    if p_adj < C.ALPHA else "Does not survive correction for the lags that were tried.",
                    ["Holm controls family-wise error only across the lags attempted here."], critical=True))

    xa, ya = S.lagged_pair(xd, yd, k)
    r_full = float(np.corrcoef(xa, ya)[0, 1])
    ci = S.block_bootstrap_ci(xa, ya, C.BOOTSTRAP_B, block, seed=C.SEED)
    ok = not (ci[0] <= 0 <= ci[1])
    tests.append(_t("block_bootstrap", "Moving-block bootstrap 95% CI",
                    {"r": r_full, "ci_95": ci, "block": block, "draws": C.BOOTSTRAP_B},
                    "CI excludes 0", "pass" if ok else "fail",
                    "Block-bootstrap interval excludes zero." if ok else "Block-bootstrap interval includes zero.",
                    ["Block length is a fixed choice; very persistent series may need longer blocks."],
                    critical=True))

    # Max-statistic circular-shift placebo (accounts for choosing the best lag).
    rng = np.random.default_rng(C.SEED)
    x_all, y_all = xd, yd
    nd = len(x_all)
    obs = max(abs(np.corrcoef(*S.lagged_pair(x_all, y_all, kk))[0, 1]) for kk in ks)
    shifts = rng.integers(30, nd - 30, C.PLACEBO_DRAWS)
    cnt = sum(max(abs(np.corrcoef(*S.lagged_pair(np.roll(x_all, s), y_all, kk))[0, 1]) for kk in ks) >= obs
              for s in shifts)
    p_pl = float((1 + cnt) / (1 + C.PLACEBO_DRAWS))
    tests.append(_t("placebo_max_stat", "Circular-shift placebo on best-of-lags statistic",
                    {"max_abs_r": float(obs), "p_placebo": p_pl, "draws": C.PLACEBO_DRAWS},
                    f"p<{C.ALPHA}", "pass" if p_pl < C.ALPHA else "fail",
                    "Stronger than the best lag of time-shifted placebo copies." if p_pl < C.ALPHA
                    else "Best-lag strength is typical of time-shifted placebo copies.",
                    ["Preserves X's autocorrelation but breaks its alignment with Y."], critical=True))

    # Stability across chronological windows.
    bounds = np.linspace(0, len(dev), C.STABILITY_WINDOWS + 1).astype(int)
    rs = [float(np.corrcoef(*_pair(dev.iloc[bounds[i]:bounds[i + 1]], k))[0, 1])
          for i in range(C.STABILITY_WINDOWS)]
    sgn = np.sign(r_full)
    same = int(sum(np.sign(v) == sgn for v in rs))
    ok = same >= C.STABILITY_MIN_SAME_SIGN
    tests.append(_t("window_stability", "Stability across chronological windows",
                    {"window_r": rs, "same_sign": same, "windows": C.STABILITY_WINDOWS},
                    f">={C.STABILITY_MIN_SAME_SIGN}/{C.STABILITY_WINDOWS} windows share the full-sample sign",
                    "pass" if ok else "fail",
                    "Effect direction is consistent across time." if ok else "Effect direction flips across time windows.",
                    ["Each window has only ~%d points." % (len(dev) // C.STABILITY_WINDOWS)]))

    # Outlier sensitivity (winsorising).
    lo, hi = C.OUTLIER_WINSOR, 1 - C.OUTLIER_WINSOR
    xw = np.clip(xa, *np.quantile(xa, [lo, hi]))
    yw = np.clip(ya, *np.quantile(ya, [lo, hi]))
    r_w, p_w = S.hac_corr(xw, yw, hac_lags)
    rel = abs(r_w - r_full) / abs(r_full) if r_full else float("inf")
    ok = rel <= C.OUTLIER_MAX_REL_CHANGE and p_w < C.ALPHA and np.sign(r_w) == sgn
    tests.append(_t("outlier_sensitivity", "Winsorised (1%/99%) re-estimate",
                    {"r": r_full, "r_winsorised": r_w, "relative_change": float(rel), "p_hac_winsorised": p_w},
                    f"change<={C.OUTLIER_MAX_REL_CHANGE:.0%} and still significant", "pass" if ok else "fail",
                    "Result is not driven by extreme days." if ok else "Result changes materially without extreme days.",
                    ["Winsorising removes genuine tail events too."]))

    # Direction consistency across the three chronological splits.
    r_split = [r_tr, r_v, r_h]
    ok = all(np.sign(v) == np.sign(r_tr) for v in r_split)
    tests.append(_t("direction_consistency", "Effect direction across train/validation/holdout",
                    {"r_train": r_tr, "r_validation": r_v, "r_holdout": r_h},
                    "all three share the same sign", "pass" if ok else "fail",
                    "Sign agrees in all splits." if ok else "Sign differs between splits.",
                    ["Sign agreement is weak evidence on its own."]))

    v, why = verdict_from_tests(tests)

    return {
        "tests": tests, "reliability_verdict": v, "verdict_rule": why,
        "strength": _ST.reliability_strength(tests),
        "verdict_criteria": {"rejected": "weighted score < 45, or the holdout fails and the score is < 70",
                             "supported": "weighted score >= 75 and the holdout passes",
                             "inconclusive": "otherwise, or n<%d" % C.MIN_VALIDATION_N,
                             "critical_tests": [t["id"] for t in tests if t["critical"]]},
        "holdout_evaluations": 1,
        "claim_tested": (f"X lagged by {k} day(s) ({'pre-registered lag' if fixed_lag is not None else 'lag chosen on train only'}) "
                         "is linearly associated with Y."),
        "selected_lag": k,
        "split": {"train": [str(train.index[0].date()), str(train.index[-1].date()), len(train)],
                  "validation": [str(val.index[0].date()), str(val.index[-1].date()), len(val)],
                  "holdout": [str(hold.index[0].date()), str(hold.index[-1].date()), len(hold)]},
        "reasoning": "",
    }
