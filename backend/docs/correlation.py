"""Stage 1: is there an association?"""
import numpy as np
from . import criteria as C, stats as S


def run_correlation(df, x_label, y_label):
    n = len(df)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    r, p_iid = S.pearson(x, y)
    r_hac, p_hac = S.hac_corr(x, y)
    rho, p_rho = S.spearman(x, y)
    rx = np.corrcoef(x[:-1], x[1:])[0, 1]
    ry = np.corrcoef(y[:-1], y[1:])[0, 1]
    n_eff = float(n * (1 - rx * ry) / (1 + rx * ry))  # Bartlett effective n
    adf_x, adf_y = S.adf_pvalue(x), S.adf_pvalue(y)

    if n < C.MIN_N:
        verdict, why = "inconclusive", f"n={n} < {C.MIN_N}."
    elif p_hac < C.ASSOC_DETECT_P and abs(r) >= C.ASSOC_DETECT_ABS_R and np.sign(rho) == np.sign(r):
        verdict, why = "detected", (f"HAC p={p_hac:.3g} < {C.ASSOC_DETECT_P}, |r|={abs(r):.3f} >= "
                                    f"{C.ASSOC_DETECT_ABS_R}, Spearman agrees in sign.")
    elif p_hac >= C.ASSOC_NULL_P:
        verdict, why = "not_detected", f"HAC p={p_hac:.3g} >= {C.ASSOC_NULL_P}."
    else:
        verdict, why = "inconclusive", "Evidence is between the detection and null criteria."

    return {
        "x_label": x_label, "y_label": y_label, "method": "pearson",
        "r": r, "p_value": p_iid, "p_value_hac": p_hac, "ci_95": S.fisher_ci(r, n),
        "n": n, "n_effective": n_eff,
        "spearman_rho": rho, "spearman_p": p_rho,
        "association_verdict": verdict, "verdict_rule": why,
        "criteria": {"detected": f"HAC p<{C.ASSOC_DETECT_P} and |r|>={C.ASSOC_DETECT_ABS_R} and Spearman same sign",
                     "not_detected": f"HAC p>={C.ASSOC_NULL_P}", "min_n": C.MIN_N},
        "assumptions": ["p_value / ci_95: i.i.d. bivariate-normal approximation (Fisher z).",
                        "p_value_hac: Newey-West errors (maxlags=%d), robust to serial dependence." % S.HAC_LAGS,
                        "Pearson r measures linear association only; it is not evidence of causality."],
        "checks": {"adf_p_x": adf_x, "adf_p_y": adf_y,
                   "stationary_at_5pct": bool(adf_x < 0.05 and adf_y < 0.05),
                   "lag1_autocorr_x": float(rx), "lag1_autocorr_y": float(ry)},
    }


def build_viz_series(df, window=C.ROLLING_WINDOW):
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    dates = [d.date().isoformat() for d in df.index]
    rr = S.rolling_r(x, y, window)
    return {
        "aligned_series": [{"date": d, "x": float(a), "y": float(b), "factor": float(f)}
                           for d, a, b, f in zip(dates, x, y, df["factor"])],
        "scatter": [{"x": float(a), "y": float(b)} for a, b in zip(x, y)],
        "rolling_correlation": [{"date": d, "r": r} for d, r in zip(dates, rr)],
        "rolling_window": window,
    }
