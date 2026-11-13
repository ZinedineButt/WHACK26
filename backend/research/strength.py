"""Graded evidence strength (0-100) next to the verdicts.

A pass/fail answer hides the case where almost everything agrees and one check wobbles. Each check is weighted, a pass counts 1,
an unclear result counts 0.5 and a fail counts 0; checks that could not run are left out. The weights are fixed here, not tuned
per run. Verdicts use the same scores, so the label and the score always agree."""

CAUSAL_WEIGHTS = {"lead_lag": 2, "confounder_control": 3, "granger": 2, "placebo_shift": 2, "negative_control": 1, "event_study": 1}
HOLDOUT_WEIGHT = 4          # the locked-away final quarter is the most honest single test
CRITICAL_WEIGHT = 2
SOFT_WEIGHT = 1
CREDIT = {"pass": 1.0, "inconclusive": 0.5, "unknown": 0.5, "fail": 0.0}

# verdict cut-offs on the 0-100 score
CAUSAL_PLAUSIBLE_MIN = 75   # and the confounder control must not fail
RELIABLE_MIN = 75           # and the holdout must not fail
UNRELIABLE_MAX = 45         # below this the evidence is rejected outright
HOLDOUT_FAIL_REJECT_BELOW = 70   # a failed holdout is only forgiven when everything else is strong


def label(pct):
    if pct is None:
        return "not assessed"
    if pct >= 80:
        return "strong"
    if pct >= 60:
        return "moderate"
    if pct >= 40:
        return "mixed"
    if pct >= 20:
        return "weak"
    return "very weak"


def _score(items):
    """items: [(weight, result)]. Returns (pct or None, passed, ran)."""
    ran = [(w, r) for w, r in items if r in CREDIT]
    if not ran:
        return None, 0, 0
    tot = sum(w for w, _ in ran)
    pct = round(100 * sum(w * CREDIT[r] for w, r in ran) / tot)
    return pct, sum(1 for _, r in ran if r == "pass"), len(ran)


def causal_strength(tests):
    pct, ok, ran = _score([(CAUSAL_WEIGHTS.get(t["id"], 1), t["result"]) for t in tests])
    return {"pct": pct, "label": label(pct), "checks_passed": ok, "checks_run": ran}


def reliability_strength(tests):
    items = []
    for t in tests:
        w = HOLDOUT_WEIGHT if t["id"] == "holdout_single_evaluation" else CRITICAL_WEIGHT if t.get("critical") else SOFT_WEIGHT
        items.append((w, t["result"]))
    pct, ok, ran = _score(items)
    return {"pct": pct, "label": label(pct), "checks_passed": ok, "checks_run": ran}


# |r| -> size score (0-1). Daily-return benchmarks: <0.1 negligible, 0.1-0.2 weak, 0.2-0.3 modest, 0.3-0.5 strong, >=0.5 exceptional.
R_SCALE = [(0.0, 0.0), (0.1, 0.2), (0.2, 0.45), (0.3, 0.7), (0.4, 0.88), (0.5, 1.0)]


def _size(a):
    if a >= R_SCALE[-1][0]:
        return 1.0
    for (x0, y0), (x1, y1) in zip(R_SCALE, R_SCALE[1:]):
        if a <= x1:
            return y0 + (y1 - y0) * (a - x0) / (x1 - x0)
    return 1.0


def association_strength(c):
    """Size of the link first, then how sure we are it is not luck. A tiny link can be statistically certain and still be weak,
    so certainty only scales the size score down; it never lifts it. Size is a piecewise-linear curve tuned to daily financial returns,
    where r = 0.2 is already notable and r >= 0.5 is exceptional (see R_SCALE); a straight line would overrate r = 0.4."""
    a, p = abs(c["r"]), c["p_value_hac"]
    size = _size(a)
    sure = 1.0 if p < 0.001 else 0.9 if p < 0.01 else 0.7 if p < 0.05 else 0.3 if p < 0.2 else 0.0
    pct = round(100 * size * sure * (1.0 if c.get("direction_consistent", True) else 0.5))
    return {"pct": pct, "label": label(pct)}


def overall(h):
    """One number for the whole hypothesis: association 20, cause 15, unseen data 40, trading 15, similar assets 10."""
    parts = []
    c, ca, v, bt, g = h.get("correlation"), h.get("causality"), h.get("validation"), h.get("backtest"), h.get("generalisation")
    if c:
        parts.append((20, c["strength"]["pct"] if c.get("strength") else None))
    if ca and ca.get("strength"):
        parts.append((15, ca["strength"]["pct"]))
    if v and v.get("strength"):
        parts.append((40, v["strength"]["pct"]))
    if bt and bt.get("applicable"):
        parts.append((15, {"pass": 100, "inconclusive": 50, "fail": 0}.get(bt.get("verdict"))))
    if g and g.get("status") == "completed":
        parts.append((10, {"generalises": 100, "partial": 60, "does_not_generalise": 0}.get(g.get("verdict"))))
    parts = [(w, p) for w, p in parts if p is not None]
    if not parts:
        return {"pct": None, "label": "not assessed"}
    pct = round(sum(w * p for w, p in parts) / sum(w for w, _ in parts))
    return {"pct": pct, "label": label(pct)}
