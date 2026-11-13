"""Core statistics. Every number in a report comes from these functions."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

HAC_LAGS = 5


def lagged_pair(x, y, k):
    """Pair x_{t-k} with y_t. k>0: x leads y. k<0: y leads x."""
    n = len(x)
    if k >= 0:
        return x[: n - k], y[k:]
    return x[-k:], y[: n + k]


def pearson(x, y):
    r, p = sps.pearsonr(x, y)
    return float(r), float(p)


def fisher_ci(r, n, level=0.95):
    if n <= 3 or abs(r) >= 1:
        return None
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    q = sps.norm.ppf(0.5 + level / 2)
    return [float(np.tanh(z - q * se)), float(np.tanh(z + q * se))]


def hac_corr(x, y, maxlags=HAC_LAGS):
    """Slope of standardised y on standardised x == r; p-value uses Newey-West
    (HAC) errors so serial dependence does not inflate significance."""
    xs = (x - x.mean()) / x.std(ddof=0)
    ys = (y - y.mean()) / y.std(ddof=0)
    res = sm.OLS(ys, sm.add_constant(xs)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.params[1]), float(res.pvalues[1])


def holm(pvals, alpha=0.05):
    """Holm step-down. Returns (adjusted p-values, reject flags)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist(), (adj < alpha).tolist()


def circular_shift_placebo(x, y, k, B=499, seed=0, min_shift=30):
    """Placebo p-value: circularly shift x (preserves its autocorrelation) and
    recompute |r| at lag k. Valid for dependent data, unlike i.i.d. permutation."""
    rng = np.random.default_rng(seed)
    n = len(x)
    xa, ya = lagged_pair(x, y, k)
    r_obs = abs(np.corrcoef(xa, ya)[0, 1])
    shifts = rng.integers(min_shift, n - min_shift, B)
    cnt = 0
    for s in shifts:
        xs, ys = lagged_pair(np.roll(x, s), y, k)
        if abs(np.corrcoef(xs, ys)[0, 1]) >= r_obs:
            cnt += 1
    return float((1 + cnt) / (1 + B))


def block_bootstrap_ci(xa, ya, B=1000, block=20, seed=0, level=0.95):
    """Moving-block bootstrap CI for r (jointly resamples (x,y) blocks)."""
    rng = np.random.default_rng(seed)
    n = len(xa)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, (B, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    X, Y = xa[idx], ya[idx]
    Xc, Yc = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    r = (Xc * Yc).sum(1) / np.sqrt((Xc**2).sum(1) * (Yc**2).sum(1))
    a = (1 - level) / 2
    return [float(np.quantile(r, a)), float(np.quantile(r, 1 - a))]


def rolling_r(x, y, window=60):
    out = [None] * len(x)
    for i in range(window - 1, len(x)):
        xs, ys = x[i - window + 1 : i + 1], y[i - window + 1 : i + 1]
        if xs.std() > 0 and ys.std() > 0:
            out[i] = float(np.corrcoef(xs, ys)[0, 1])
    return out


def adf_pvalue(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(sm.tsa.adfuller(s, autolag="AIC")[1])


def spearman(x, y):
    rho, p = sps.spearmanr(x, y)
    return float(rho), float(p)


def align(series_x: pd.Series, series_y: pd.Series, *extra):
    """Inner-join on timestamp, drop rows with NaN/inf. Returns frame + drop count."""
    df = pd.concat({"x": series_x, "y": series_y, **{f"e{i}": e for i, e in enumerate(extra)}}, axis=1, join="inner")
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, before - len(df)
