"""Hypothesis generation + narrative. Gemini is used ONLY for structured
explanations and prose; all statistics come from Python. Without a key (or if
Gemini fails) deterministic templates are used and clearly labelled."""
import json
import os
import re
import time
from typing import List, Literal

from pydantic import BaseModel, ValidationError

SOURCE_GEMINI = "gemini"
SOURCE_TEMPLATE = "template_demo"           # no key configured
SOURCE_FALLBACK = "template_fallback"       # key configured but Gemini failed / invalid output

RETRIES = 4
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class Explanation(BaseModel):
    id: str
    title: str
    type: Literal["direct_effect", "common_cause", "reverse_causation", "coincidence"]
    description: str


class Confounder(BaseModel):
    name: str
    rationale: str
    observed: bool


class Hypotheses(BaseModel):
    candidate_explanations: List[Explanation]
    possible_confounders: List[Confounder]


class Narrative(BaseModel):
    causality_reasoning: str
    validation_reasoning: str


def template_hypotheses(ctx):
    x, y = ctx["x_label"], ctx["y_label"]
    return {
        "candidate_explanations": [
            {"id": "direct", "type": "direct_effect", "title": "X influences Y",
             "description": f"Moves in {x} could feed into {y}, possibly with a delay."},
            {"id": "common", "type": "common_cause", "title": "A shared driver moves both",
             "description": "A common market-wide factor could move both series at the same time without either affecting the other."},
            {"id": "reverse", "type": "reverse_causation", "title": "Y influences X",
             "description": f"{y} could lead {x} instead of the reverse."},
            {"id": "chance", "type": "coincidence", "title": "Coincidence",
             "description": "In a finite sample two unrelated series can look correlated by chance."},
        ],
        "possible_confounders": [
            {"name": "Market factor", "observed": True,
             "rationale": "Broad market moves typically drive many stock returns at once; an observed market-factor series is available."},
            {"name": "Volatility regime / news flow", "observed": False,
             "rationale": "Unobserved shared shocks cannot be controlled for with the available data."},
        ],
    }


def _numbers(obj, acc):
    """Collect every numeric fact (as float) that a narrative may legitimately quote."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, acc)


def unsupported_numbers(text, facts):
    """Decimal numbers in `text` that match no computed fact (tolerates rounding)."""
    allowed = set()
    _numbers(facts, allowed)
    bad = []
    for m in re.findall(r"\d+\.\d+", text):
        v = float(m)
        if not any(abs(abs(v) - abs(a)) <= max(0.0051, 0.005 * abs(a)) for a in allowed):
            bad.append(m)
    return bad


def template_narrative(f):
    c, ca, v = f["correlation"], f["causality"], f["validation"]
    caus = (f"Pearson r between {c['x_label']} and {c['y_label']} is {c['r']:.3f} (n={c['n']}), "
            f"verdict '{c['association_verdict']}'. Causal verdict: '{ca['causal_verdict']}'. {ca['verdict_rule']} "
            "Checks are necessary but not sufficient; unobserved confounders cannot be excluded.")
    n_pass = sum(t["result"] == "pass" for t in v["tests"])
    val = (f"The sceptic ran {len(v['tests'])} tests and {n_pass} passed; reliability verdict "
           f"'{v['reliability_verdict']}'. {v['verdict_rule']} Reliability of an association is not evidence of causation.")
    return {"causality_reasoning": caus, "validation_reasoning": val}


class Reasoner:
    def __init__(self, api_key=None, model=None, client=None, timeout_s=30):
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self._client = client

    @property
    def configured(self):
        return bool(self._client or self.api_key)

    def _call(self, prompt, schema):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        from google.genai import types
        cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                          response_schema=schema, temperature=0.2)
        for attempt in range(RETRIES):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                return json.loads(resp.text)
            except Exception as e:  # retry only transient provider errors
                transient = any(k in f"{type(e).__name__} {e}" for k in ("503", "429", "UNAVAILABLE", "DEADLINE", "overloaded"))
                if not transient or attempt == RETRIES - 1:
                    raise
                time.sleep(1.5 * 2 ** attempt)

    def hypotheses(self, ctx):
        """ctx must NOT contain ground truth. Returns (data, source, warnings)."""
        if not self.configured:
            return template_hypotheses(ctx), SOURCE_TEMPLATE, []
        prompt = (
            "You are a sceptical quantitative researcher. Two daily return series were found to be "
            f"associated: X = '{ctx['x_label']}', Y = '{ctx['y_label']}', Pearson r = {ctx['r']:.3f}, n = {ctx['n']}. "
            "An observed market-factor series is available as a control. Propose 3-5 candidate explanations "
            "(must include at least one non-causal alternative: common_cause, reverse_causation or coincidence) "
            "and plausible confounders. Do not claim any explanation is true; do not invent numbers.")
        try:
            data = Hypotheses.model_validate(self._call(prompt, Hypotheses)).model_dump()
            types_ = {e["type"] for e in data["candidate_explanations"]}
            if len(data["candidate_explanations"]) < 2 or types_ == {"direct_effect"}:
                raise ValueError("no alternative explanation proposed")
            return data, SOURCE_GEMINI, []
        except Exception as e:  # network, quota, schema, safety ...
            return template_hypotheses(ctx), SOURCE_FALLBACK, [f"Gemini hypothesis generation failed ({type(e).__name__}); template used."]

    def narrative(self, facts):
        if not self.configured:
            return template_narrative(facts), SOURCE_TEMPLATE, []
        prompt = (
            "Write two short paragraphs (<=90 words each) for a research report using ONLY the JSON facts below. "
            "Never say a relationship is 'proven'. Do not introduce any number that is not in the facts. "
            "Return fields causality_reasoning and validation_reasoning.\nFACTS:\n" + json.dumps(facts))
        try:
            data = Narrative.model_validate(self._call(prompt, Narrative)).model_dump()
            text = data["causality_reasoning"] + " " + data["validation_reasoning"]
            bad = unsupported_numbers(text, facts)
            if bad:
                raise ValueError(f"narrative contains unsupported numbers {bad[:3]}")
            if re.search(r"\bprov(en|es|e)\b", text, re.I):
                raise ValueError("narrative claims proof")
            return data, SOURCE_GEMINI, []
        except Exception as e:
            return template_narrative(facts), SOURCE_FALLBACK, [f"Gemini narrative failed ({type(e).__name__}); template used."]
