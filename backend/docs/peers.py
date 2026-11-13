"""Cross-asset generalisation: re-run the SAME frozen hypothesis (same predictor,
transforms, lag and sign) on peer assets chosen by a fixed server-side map.
Gemini cannot influence the peer list. Peers are only tested when the primary
association was detected (rule declared in the pre-registration)."""
import numpy as np

from . import backtest as BT, criteria as C, hstages as H, registry as R, stats as S
from .providers import ProviderError

# Fixed, editable peer groups. Order = priority.
PEER_GROUPS = {
    "us_equity_broad": ["SPY", "QQQ", "IWM", "DIA", "SP500", "NASDAQCOM", "NASDAQ100", "DJIA"],
    "technology": ["QQQ", "XLK", "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "NASDAQCOM", "NASDAQ100"],
    "energy": ["XLE", "XOM"],
    "airlines": ["JETS", "DAL", "UAL"],
    "financials": ["XLF", "JPM"],
    "treasuries": ["TLT"],
    "gold": ["GLD"],
    "synthetic_airlines": ["SYN_AIR", "SYN_AIR2", "SYN_TECH", "SYN_MKT"],
    "synthetic_equities": ["SYN_TECH", "SYN_MKT", "SYN_AIR2"],
}


def candidates(outcome_id, predictor_id, reg: R.Registry, cap=C.PEER_MAX):
    """Peer ids for an outcome: other members of every group containing it (priority order)."""
    out = []
    groups = [g for g, m in PEER_GROUPS.items() if outcome_id in m]
    for g in groups:
        for m in PEER_GROUPS[g]:
            if m in (outcome_id, predictor_id) or m in out:
                continue
            v, why = reg.resolve(m)
            if v is not None and not why:
                out.append(m)
    return out[:cap]


def run_peers(h, peer_ids, svc, lookback, levels, predictor_sign):
    """h: normalized hypothesis. Returns dict (never raises)."""
    pid = h["predictor"]["variable_id"]
    rows, skipped = [], []
    k, L = h["lag_days"], H.hac_lags_for(h)
    for peer in peer_ids:
        try:
            v = svc.registry.resolve(peer)[0]
            if peer not in levels:
                levels[peer] = svc.data.get(v, lookback)
            hp = {**h, "outcome": {**h["outcome"], "variable_id": peer, "label": v.label}}
            df, _ = H.build_frame(hp, {pid: levels[pid][0], peer: levels[peer][0]}, {})
        except (ProviderError, ValueError, KeyError) as e:
            skipped.append({"variable_id": peer, "reason": str(e)[:160]})
            continue
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        xa, ya = S.lagged_pair(x, y, k)
        r, p = S.hac_corr(xa, ya, L)
        row = {"variable_id": peer, "label": v.label, "n": int(len(xa)), "r": r, "p_hac": p,
               "same_sign_as_primary": bool(np.sign(r) == predictor_sign)}
        if h["outcome"]["transform"] == R.LOG_RETURN:
            row["backtest"] = BT.peer_metrics(x, y, k, predictor_sign)
        rows.append(row)
    if not rows:
        return {"status": "not_run", "verdict": "not_run", "peers": [], "skipped": skipped,
                "verdict_rule": "No peer data could be retrieved."}
    adj, rej = S.holm([r["p_hac"] for r in rows], C.ALPHA)
    for r_, a, j in zip(rows, adj, rej):
        r_["p_holm"], r_["significant_after_holm"] = a, bool(j)
    share_sign = float(np.mean([r_["same_sign_as_primary"] for r_ in rows]))
    share_sig = float(np.mean([r_["significant_after_holm"] and r_["same_sign_as_primary"] for r_ in rows]))
    if share_sign >= C.PEER_SAME_SIGN_SHARE and share_sig >= C.PEER_SIGNIFICANT_SHARE:
        verdict = "generalises"
    elif share_sig > 0 or share_sign >= C.PEER_SAME_SIGN_SHARE:
        verdict = "partial"
    else:
        verdict = "does_not_generalise"
    return {
        "status": "completed", "verdict": verdict, "peers": rows, "skipped": skipped, "n_peers": len(rows),
        "share_same_sign": share_sign, "share_significant_same_sign": share_sig, "correction": "holm (over peers)",
        "verdict_rule": (f"{share_sign:.0%} of peers share the primary sign; {share_sig:.0%} are significant in that "
                         f"direction after Holm. generalises needs >= {C.PEER_SAME_SIGN_SHARE:.0%} and >= {C.PEER_SIGNIFICANT_SHARE:.0%}."),
        "note": "Frozen rule re-applied to the peer's full sample; peers were never used to choose the hypothesis or lag.",
    }
