"""Hypothesis schema + server-side validation of model proposals.

Nothing from a model is ever executed: a proposal is just data that must map
onto catalogue variables, allowlisted transforms and a fixed test type."""
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from . import registry as R

MAX_LAG = 5
MAX_CONTROLS = 3
PROPOSED_TEST = "lagged_correlation_hac"
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class HypothesisProposal(BaseModel):
    hypothesis_id: str = ""
    question: str
    predictor_variable: str
    predictor_transform: str
    outcome_variable: str
    outcome_transform: str
    direction: Literal["positive", "negative", "any"]
    lag_days: int
    null_hypothesis: str
    mechanism: str
    competing_explanations: List[str]
    control_variables: List[str] = Field(default_factory=list)
    proposed_test: str = PROPOSED_TEST
    limitations: List[str] = Field(default_factory=list)


class GenerationEnvelope(BaseModel):
    hypotheses: List[HypothesisProposal]
    unsupported_reason: str = ""


def clean(text, limit):
    return _CTRL.sub("", str(text)).strip()[:limit]


def validate_proposal(p: HypothesisProposal, reg: R.Registry, allowed_ids: Optional[set], fixtures_ok: bool):
    """Returns (normalized dict | None, reasons[] (rejections), notes[] (normalisations))."""
    reasons, notes = [], []
    pv, why = reg.resolve(p.predictor_variable)
    ov, why2 = reg.resolve(p.outcome_variable)
    for role, v, w, raw in (("predictor", pv, why, p.predictor_variable), ("outcome", ov, why2, p.outcome_variable)):
        if v is None or w:
            reasons.append(f"{role} '{clean(raw, 40)}' not usable: {w}")
        elif v.source == "fixture" and not fixtures_ok:
            reasons.append(f"{role} '{v.id}' is synthetic and not allowed in live modes")
        elif v.source != "fixture" and fixtures_ok:
            reasons.append(f"{role} '{v.id}' is not a fixture variable (demo mode is synthetic only)")
        elif allowed_ids is not None and v.id not in allowed_ids:
            reasons.append(f"{role} '{v.id}' is outside the requested universe")
    if reasons:
        return None, reasons, notes
    if pv.id == ov.id:
        reasons.append("predictor and outcome are the same variable")
    pt, ot = p.predictor_transform, p.outcome_transform
    if pt not in pv.transforms or pt in R.FORWARD_OUTCOMES:
        reasons.append(f"transform '{clean(pt, 30)}' not allowed for predictor {pv.id} (allowed: "
                       f"{[t for t in pv.transforms if t not in R.FORWARD_OUTCOMES]})")
    if ot not in ov.transforms:
        reasons.append(f"transform '{clean(ot, 30)}' not allowed for outcome {ov.id} (allowed: {list(ov.transforms)})")
    if p.proposed_test != PROPOSED_TEST:
        reasons.append(f"unsupported test '{clean(p.proposed_test, 40)}' (only {PROPOSED_TEST})")
    lag = p.lag_days
    if ot in R.FORWARD_OUTCOMES:
        if lag != 0:
            notes.append(f"lag_days {lag} -> 0: forward-window outcome '{ot}' already starts after the predictor date")
        lag = 0
    elif not 1 <= lag <= MAX_LAG:
        reasons.append(f"lag_days must be 1..{MAX_LAG} (predictive, not contemporaneous); got {lag}")
    if len(p.question.strip()) < 10:
        reasons.append("question too short")
    if not p.null_hypothesis.strip() or not p.mechanism.strip():
        reasons.append("null_hypothesis and mechanism are required")
    controls = []
    for c in p.control_variables[:MAX_CONTROLS + 3]:
        cv, cw = reg.resolve(c)
        ok_fix = cv is not None and ((cv.source == "fixture") == fixtures_ok)
        if cv is None or cw or not ok_fix or cv.id in (pv.id, ov.id) or cv.default_transform is None or cv.id in controls:
            notes.append(f"control '{clean(c, 30)}' dropped (unusable, duplicate or same as predictor/outcome)")
        elif len(controls) < MAX_CONTROLS:
            controls.append(cv.id)
        else:
            notes.append(f"control '{cv.id}' dropped (max {MAX_CONTROLS})")
    if reasons:
        return None, reasons, notes
    return {
        "question": clean(p.question, 400),
        "predictor": {"variable_id": pv.id, "label": pv.label, "transform": pt},
        "outcome": {"variable_id": ov.id, "label": ov.label, "transform": ot},
        "direction": p.direction, "lag_days": lag,
        "null_hypothesis": clean(p.null_hypothesis, 400), "mechanism": clean(p.mechanism, 700),
        "competing_explanations": [clean(x, 300) for x in p.competing_explanations[:5]],
        "control_variables": controls, "proposed_test": PROPOSED_TEST,
        "limitations": [clean(x, 300) for x in p.limitations[:5]],
        "required_datasets": sorted({pv.id, ov.id, *controls}),
    }, [], notes
